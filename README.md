# JARVIS — a voice AI assistant (local-first, cloud-optional)

A voice-enabled assistant in the spirit of Iron Man's JARVIS. Wake it with
**"Hey Jarvis,"** ask general questions, and control apps — Spotify for music and
Chrome for the browser/web. It runs **fully local by default**, or can switch to a
cloud pipeline.

The brain is a **LangGraph** react-style agent (all tools bound to one local model)
that also **remembers you over time**. It answers general facts itself, controls
Spotify, drives Chrome, and searches the web:

- *"play despacito"*, *"pause"*, *"play my liked songs"* → Spotify.
- *"open instagram"*, *"play some lofi on youtube"*, *"open a new MrBeast video"*,
  *"show me some reels"* → Chrome.
- *"what's the latest news on X"*, *"current price of bitcoin"* → web search
  (Tavily). Set `TAVILY_API_KEY` in `.env`.
- *"download my OS study materials by Prof X from VTOP"*, *"check my AWS balance"* →
  the **browser agent** ([browser-use](https://github.com/browser-use/browser-use))
  drives your **real logged-in Chrome** and does the multi-step workflow, then
  speaks the result. One-time setup: `bash scripts/setup_browser_agent.sh`.

  The browser agent runs in an **isolated venv** (`.venv-browser`, invoked
  out-of-process) because its deps conflict with the LiveKit/LangGraph stack. It
  uses your **real Chrome profile** (saved logins/passwords), so **Chrome must be
  closed** when a task runs — JARVIS will ask you to close it otherwise. Reasoning
  uses a frontier model (`gpt-4.1-mini` via `OPENAI_API_KEY`) because local models
  can't drive a browser reliably. ⚠️ Two conscious tradeoffs: **browsed page
  content goes to OpenAI**, and with **full autonomy on your logged-in profile** a
  malicious page could prompt-inject the agent — it runs only when you ask, with
  step/time caps, but no per-action confirmations. It fills forms, clicks, uploads
  files, logs in (via your saved sessions/passwords), and fetches info. If a
  **captcha** blocks a task, it auto-solves distorted-text/image captchas with a
  **local open-source vision model** (Qwen2.5-VL via Ollama — free, private, no
  service): `ollama pull qwen2.5vl:7b` (toggle `JARVIS_BROWSER_CAPTCHA`). It runs
  headless on a **clone of your Chrome profile** so it works even while your Chrome
  is open (`JARVIS_BROWSER_CLONE_PROFILE=auto`).
- *"download the latest assignment from VTOP and explain it"* → *"finish it, make it
  a jupyter notebook"* → *"open it"* → the **assignment workflow**: JARVIS downloads
  the document (browser agent), **reads and explains it aloud** (local model), then
  hands it to an **AI to complete** it, **assembles the answer file** in the format
  you asked for (`.ipynb`/`.docx`/`.pptx`/`.py`), announces it's done, and **opens it**
  in the right app (Preview / Word / PowerPoint / VS Code).

  Which AI does the work is chosen by availability, in your preferred order
  (`JARVIS_AI_APP_ORDER`, default `claude_desktop,browser`). The **Claude desktop app**
  is used only if it's installed, macOS **Accessibility** is granted to JARVIS, and
  `JARVIS_CLAUDE_DESKTOP=1` (it's Electron UI-automation — experimental, and answer
  extraction is unreliable); otherwise it **falls back to the browser** (claude.ai, your
  logged-in Pro session, driven by the browser agent). The assignment content goes to
  whichever AI completes it; the explain step is local.

**Persistent memory (GPT-style, fully local).** JARVIS learns what you like from
what you do (songs played, sites/searches), from explicit *"remember that I…"*, and
from durable facts mined out of conversation in the background. Relevant memories
are recalled each turn to personalise replies — ask *"play me something I'd enjoy"*
or *"what do you know about me?"*, and *"forget that…"* to remove one. Memories live
in `~/.jarvis/memory/` (local JSON + Ollama embeddings); nothing leaves the machine.
Needs an embedding model: `ollama pull nomic-embed-text`.

Why one agent with all tools rather than multi-agent handoffs? On local models,
agent handoffs are slow and make the model speak plumbing / narrate instead of
calling tools (measured). A single shallow graph is fast (~2–4s/action) and
reliable. Set `JARVIS_ORCHESTRATOR=native` to fall back to the previous
hand-rolled agent.

## Two modes (`JARVIS_MODE`)

| | `0` LOCAL (default) | `1` CLOUD |
|---|---|---|
| STT | Whisper mlx (`small.en`) | Deepgram `nova-3` |
| LLM | Ollama `qwen2.5:7b-instruct` | Cerebras `gpt-oss-120b` → OpenAI |
| TTS | Kokoro-82M | Deepgram Aura-2 (`aura-2-draco-en`) |

LOCAL keeps everything on-device — nothing the mic captures leaves the machine
(the only egress is the optional Spotify catalog search, a song title). CLOUD
mirrors the reference "live JARVIS" stack; the LLM prefers Cerebras and falls
back to OpenAI based on which API key is set.

## Architecture

Voice pipeline (all local):
```
mic ─► Silero VAD ─► Whisper STT ─► LangGraph brain (Ollama + memory) ─► Kokoro TTS ─► speaker
```

Activation: **wake word with smart follow-up**.
```
mic ─► VAD/turn-detect ─► STT ─► [wake gate] ─► LLM ─► TTS ─► speaker
                                  awake?  ── no ──► ignore
```
Say **"jarvis"** or **"hey jarvis"** to activate (the wake word is stripped, so
"jarvis play some jazz" just plays jazz). The clever bit:

- If JARVIS's reply is a **clarification question** ("Which playlist?"), it stays
  awake — you answer directly, no wake word.
- If JARVIS's reply is an **answer/statement**, it goes back to sleep — say the
  wake word again for the next request.

Transcript-based (so both "jarvis" and "hey jarvis" work). Configure with
`JARVIS_WAKE_WORDS` (comma-separated), `JARVIS_WAKE_FOLLOWUP` (window seconds), or
disable with `JARVIS_WAKE=0` to reply to everything.

### Performance / quantization
All local LLMs run **Q4_K_M** (4-bit quantized) weights via Ollama. For faster
inference, enable Flash Attention and a **quantized KV cache** on the Ollama
server (big win on Apple Silicon, less memory per token):
```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0   # or q4_0 for even less memory
brew services restart ollama
```
STT uses `fp16` mlx-whisper on Metal. Want lighter still? Point a model env at a
lower-bit tag, e.g. `JARVIS_RELEVANCE_MODEL=qwen3:1.7b-q4_0`.

### Model (local)
The local LLM is **`qwen2.5:7b-instruct`** (`OLLAMA_MODEL`). It's non-thinking, so
tool calls are immediate — Qwen3 is a *reasoning* model that emits a 10–40s
"thinking" trace before every tool call (unusable for voice). 7B is reliable at
tool-calling on arbitrary songs; drop to `qwen2.5:3b-instruct` for speed (less
reliable) or up for more quality.

### One agent, not a mesh
JARVIS is a **single agent** that answers questions and controls Spotify. An
earlier router+specialist handoff design was removed: with local models, agent
handoffs were unreliable — after a handoff the model would emit the tool call as
plain text instead of executing it. Two hard-won lessons baked in here:
- **Single agent + all tools** executes reliably; handoffs don't (locally).
- **Keep the system prompt lean and behavioral.** Enumerating tools/JSON in the
  prompt makes local models *narrate* the call ("now playing X") without actually
  calling the tool. The tool docstrings describe usage; the prompt just sets
  behavior.

### Spotify capabilities
Playback runs **in the background** — Spotify is launched hidden (`open -g -j`) and
driven by AppleScript, and every call runs in a worker thread, so it never steals
focus or keystrokes from what you're doing.

- Search & play a song — *"play Bohemian Rhapsody by Queen"*
- Loop / repeat the current song — *"loop this"* / *"play Weightless on repeat"*
- Pause / resume / skip / volume / what's playing
- Playlists — *"what playlists do I have"*, *"play my Focus playlist"*, *"add this to my Favourites"*
- **Your library** — *"play my most played song"*, *"what are my top songs"*,
  *"list my liked songs"*, *"play one of my favourites"*, *"who are my top
  artists"*, *"what did I listen to recently"*

Song search needs the client-credentials key. **Playlist & library features**
(your playlists, top tracks, liked songs, recently played) need a one-time user
login — re-run this whenever new scopes are added:
```bash
# add http://127.0.0.1:8080/callback to your app's Redirect URIs, then:
python -m jarvis.spotify_auth
```
Scopes requested: playlist read/modify, `user-top-read`, `user-library-read`,
`user-read-recently-played`.

### Sandboxing
The only capability that touches your machine is **Spotify** (play / playlists /
loop / add-to-playlist / pause / skip / volume / now-playing). There are **no
file, shell, calendar, or delete/remove tools** exposed to any model — nothing
can delete tracks or playlists (the Spotify surface has no destructive op).
ChatAgent answers from the model's own knowledge only. Any URI interpolated into
AppleScript is regex-validated (`spotify:{track,playlist,album,artist}:…`) first,
so it can't be used for injection.

## Setup

### 1. Ollama + the LLM
```bash
brew install ollama
ollama serve            # leave running (or: brew services start ollama)
ollama pull qwen3:8b
```

### 2. Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Kokoro voice model (download once, ~350 MB)
```bash
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 4. Spotify credentials (for search only)
Create a free app at https://developer.spotify.com/dashboard, then:
```bash
cp .env.example .env
# fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
```
Make sure the **Spotify desktop app is installed and logged in** (Premium recommended).

## Run
```bash
source .venv/bin/activate
python -m jarvis.agent console
```
Say **"Hey Jarvis"**, then your request (follow-ups within ~12s skip the wake word):
*"Play Bohemian Rhapsody by Queen."* · *"Pause."* · *"What's playing?"* ·
*"What's on my calendar today?"* · *"Find my budget spreadsheet."* ·
*"What's the tallest mountain in the world?"*

Set `JARVIS_WAKE=0` for always-listening, or `JARVIS_MODE=1` (with cloud keys in
`.env`) for the cloud pipeline.

## Roadmap
- [x] Local voice loop + Spotify search/play
- [x] Multi-agent handoff mesh (router + Chat/Music/Calendar/Files)
- [x] Wake word ("Hey Jarvis") via openWakeWord
- [x] mlx-whisper (Metal) fast local STT backend
- [x] Cloud mode (Deepgram STT+TTS + Cerebras/OpenAI)
- [ ] More apps (mail, messages, system control)
- [ ] Persistent long-term memory (RAG)
