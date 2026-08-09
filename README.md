# JARVIS — a voice AI assistant (local-first, cloud-optional)

A voice-enabled assistant in the spirit of Iron Man's JARVIS. Wake it with
**"Hey Jarvis,"** ask general questions, and control apps — starting with Spotify.
It runs **fully local by default**, or can switch to a cloud pipeline.

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
mic ─► Silero VAD ─► Whisper STT ─► qwen3 (Ollama) ─► Kokoro TTS ─► speaker
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
