# JARVIS — a fully-local voice AI assistant

A completely on-device, voice-enabled assistant in the spirit of Iron Man's JARVIS.
Speech-to-text, reasoning, and text-to-speech all run locally on your Mac. It answers
general questions and controls apps — starting with Spotify (search + play).

## What stays local vs. what touches the network

- **100% local:** wake/turn detection, speech-to-text, the LLM, text-to-speech.
  Nothing the microphone captures ever leaves the machine.
- **Only network call:** the Spotify **catalog search** (we send a song title to
  Spotify's API to resolve it to a track). Playback itself is local, driven by
  AppleScript against the Spotify desktop app. Spotify is a cloud streaming service,
  so its audio comes from the network regardless — but your voice never does.

## Architecture

Voice pipeline (all local):
```
mic ─► Silero VAD ─► Whisper STT ─► qwen3 (Ollama) ─► Kokoro TTS ─► speaker
```

Multi-agent mesh (no single agent does everything — control is handed off):
```
                 ┌──────────────┐
   session ─────►│ RouterAgent  │  greets, decides intent, routes only
                 └──────┬───────┘
              transfer_to_*  │
             ┌──────────────┴──────────────┐
             ▼                             ▼
      ┌────────────┐   transfer_to_music  ┌────────────┐
      │  ChatAgent │◄────────────────────►│ MusicAgent │
      │ general Q&A│   transfer_to_chat   │  Spotify   │
      └────────────┘                      └─────┬──────┘
                                                │ tools
                                 Spotify (search: app UI / Web API,
                                          playback: AppleScript)
```
The shared `SpotifyController` lives in `session.userdata`; handoffs pass
conversation history via `chat_ctx` but reset each agent's own instructions.

| Layer         | Tech                                   |
|---------------|----------------------------------------|
| Orchestration | LiveKit Agents (console mode, local)   |
| VAD           | Silero                                 |
| STT           | faster-whisper (`base.en` default)     |
| LLM           | Ollama · `qwen3:8b` (tool calling)     |
| TTS           | Kokoro-82M (ONNX), British male voice  |

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
Then talk: *"Play Bohemian Rhapsody by Queen."* · *"Pause."* · *"What's playing?"* ·
*"What's the tallest mountain in the world?"*

## Roadmap
- [x] Phase 1: local voice loop + Spotify search/play (this)
- [ ] Wake word ("Hey Jarvis") via openWakeWord
- [ ] Multi-agent handoff (dedicated Spotify / calendar / file agents)
- [ ] More apps (calendar, mail, system control)
- [ ] STT upgrade to whisper.cpp / mlx-whisper (Metal) for lower latency
