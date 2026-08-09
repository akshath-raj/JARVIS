# JARVIS — a voice AI assistant (local-first, cloud-optional)

A voice-enabled assistant in the spirit of Iron Man's JARVIS. Wake it with
**"Hey Jarvis,"** ask general questions, and control apps — starting with Spotify.
It runs **fully local by default**, or can switch to a cloud pipeline.

## Two modes (`JARVIS_MODE`)

| | `0` LOCAL (default) | `1` CLOUD |
|---|---|---|
| STT | Whisper (`mlx`/`faster`) | Deepgram `nova-3` |
| LLM | Ollama `qwen3:8b` | Cerebras `gpt-oss-120b` → OpenAI |
| TTS | Kokoro-82M | Deepgram Aura-2 (`aura-2-draco-en`) |

LOCAL keeps everything on-device — nothing the mic captures leaves the machine
(the only egress is the optional Spotify catalog search, a song title). CLOUD
mirrors the reference "live JARVIS" stack; the LLM prefers Cerebras and falls
back to OpenAI based on which API key is set.

**Hybrid (default in cloud mode):** device-action agents (music, calendar, files)
run their tool-calling on the **local** LLM for reliable, private on-device
actions, while STT/TTS and general answering use the cloud. Requires Ollama
running; disable with `JARVIS_HYBRID_LOCAL_ACTIONS=0`.

## Architecture

Voice pipeline (all local):
```
mic ─► Silero VAD ─► Whisper STT ─► qwen3 (Ollama) ─► Kokoro TTS ─► speaker
```

JARVIS listens continuously; a relevance gate (not a wake word) decides what to
answer:
```
mic ─► VAD/turn-detect ─► STT ─► [relevance gate] ─► LLM ─► TTS ─► speaker
                                  is this directed at JARVIS?
                                  (tiny local model, context-aware) ─ no ─► stay silent
```
No "Hey Jarvis" needed by default. A small fast model looks at the conversation
history and the new utterance and decides if it's addressed to JARVIS vs.
background chatter/noise (fails open, so it never goes unexpectedly mute). Set
`JARVIS_WAKE=1` to use the "Hey Jarvis" wake word instead, or `JARVIS_RELEVANCE=0`
to reply to everything.

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

### Per-task models (smaller = faster)
Each job uses a right-sized local model, independently swappable:

| Task | Default | Env |
|---|---|---|
| Chat / routing / reasoning | `qwen3:8b` | `OLLAMA_MODEL` |
| Device tool-calling (Spotify) | `qwen2.5:3b-instruct` | `JARVIS_TOOL_MODEL` |
| Relevance gate | `qwen2.5:1.5b-instruct` | `JARVIS_RELEVANCE_MODEL` |

**Why non-thinking models for tools/relevance:** Qwen3 is a *reasoning* model — it
emits a long "thinking" trace before a tool call (seconds of latency), and a token
cap just truncates it before the call. The compact **Qwen2.5-Instruct** models are
non-thinking, so they emit tool calls immediately — far snappier for voice. For
even stronger function-calling, point `JARVIS_TOOL_MODEL` at a fine-tuned model
(Katanemo `Arch-Function-3B`, MadeAgents `Hammer2.1-3b`, Salesforce `xLAM-2-1b`;
pull via `ollama pull hf.co/<repo>`).

**Fallback:** if the compact tool model struggles with the richer toolset
(playlists, loop, add-to-playlist), bump it up: `JARVIS_TOOL_MODEL=qwen3:8b`.
Use the smaller model when it performs well; fall back to 8B only if it doesn't.

Multi-agent hub-and-spoke (no single agent does everything — control is handed off):
```
                 ┌──────────────┐
   session ─────►│ RouterAgent  │  greets, routes only
                 └──────┬───────┘
          transfer_to_* │ ▲ back_to_coordinator
             ┌──────────┴──────────┐
             ▼                     ▼
       ┌──────────┐         ┌────────────┐
       │ ChatAgent│         │ MusicAgent │
       │ Q&A only │         │  Spotify   │
       └──────────┘         └────────────┘
```
The coordinator routes; each specialist owns its own tools and hands control
**back** to the coordinator to switch domains. Shared state lives in
`session.userdata`; handoffs pass conversation history via `chat_ctx` but reset
each agent's own instructions.

### Spotify capabilities
Playback runs **in the background** — Spotify is launched hidden (`open -g -j`) and
driven by AppleScript, and every call runs in a worker thread, so it never steals
focus or keystrokes from what you're doing.

- Search & play a song — *"play Bohemian Rhapsody by Queen"*
- Loop / repeat — *"play Weightless on repeat"* (or *"loop this"*)
- Play one of your playlists — *"play my Focus playlist"*
- Add the current song to a playlist — *"add this to my Favourites"*
- Pause / resume / skip / volume / what's playing

Song search needs the client-credentials key. **Playlist features** (play/add to
*your* playlists) need a one-time user login:
```bash
# add http://127.0.0.1:8080/callback to your app's Redirect URIs, then:
python -m jarvis.spotify_auth
```

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
