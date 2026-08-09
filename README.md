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

### Per-task models (smaller = faster)
Each job uses a right-sized local model, independently swappable:

| Task | Default | Env |
|---|---|---|
| Chat / routing / reasoning | `qwen3:8b` | `OLLAMA_MODEL` |
| Device actions (music/calendar/files) | `qwen3:4b` | `JARVIS_TOOL_MODEL` |
| Relevance gate | `qwen3:1.7b` | `JARVIS_RELEVANCE_MODEL` |

All are Qwen3 (Apache-2.0, native tool-calling). For even stronger function-calling
you can point `JARVIS_TOOL_MODEL` at a fine-tuned model, e.g. Katanemo
`Arch-Function-3B`, MadeAgents `Hammer2.1-3b`, or Salesforce `xLAM-2-1b`
(pull via `ollama pull hf.co/<repo>` — ensure the GGUF ships a tool template).

Multi-agent hub-and-spoke (no single agent does everything — control is handed off):
```
                        ┌──────────────┐
        session ───────►│ RouterAgent  │  greets, routes only
                        └──────┬───────┘
                     transfer_to_* │ ▲ back_to_coordinator
          ┌───────────┬───────────┼───────────┬───────────┐
          ▼           ▼           ▼           ▼
    ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
    │ ChatAgent│ │MusicAgent│ │ Calendar │ │  Files  │
    │ Q&A      │ │ Spotify  │ │ (macOS)  │ │(Spotlight)│
    └──────────┘ └─────────┘ └──────────┘ └─────────┘
```
The coordinator routes; each specialist owns its own tools and hands control
**back** to the coordinator to switch domains. Shared state (`SpotifyController`,
wake gate) lives in `session.userdata`; handoffs pass conversation history via
`chat_ctx` but reset each agent's own instructions.

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
