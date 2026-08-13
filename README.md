# JARVIS — a voice AI assistant for macOS

A voice-enabled assistant in the spirit of Iron Man's JARVIS. Wake it with
**"Hey Jarvis,"** talk to it, and it controls your Mac: plays music, drives the
browser, reads and explains what's on your screen, answers questions from your own
documents, manages a calendar and reminders, organises your files, and runs a
focus-assist mode that closes distractions — all narrated back to you, with a
hidden **Iron-Man-style HUD** that appears on launch.

It runs **cloud-first by default** (fast, hosted models) and can also run **fully
on-device**. One agent holds every tool and chains them for multi-step requests
(*"explain this graph and check the 2026 numbers"* → looks at your screen, then
searches the web, then answers).

---

## Contents
- [Feature overview](#feature-overview)
- [Two pipelines (`JARVIS_MODE`)](#two-pipelines-jarvis_mode)
- [Quick start](#quick-start)
- [Setup & API keys](#setup--api-keys)
- [macOS permissions](#macos-permissions)
- [Configuration reference](#configuration-reference)
- [Architecture](#architecture)
- [Privacy & safety](#privacy--safety)

---

## Feature overview

Everything below is available with the default `openai` brain (a single agent on
Cerebras/OpenAI). Say the wake word, then your request.

### 🎵 Music (Spotify)
Playback runs **in the background** (launched hidden, driven by AppleScript) so it
never steals focus. Needs the Spotify desktop app installed + logged in.
- *"play Bohemian Rhapsody by Queen"*, *"pause"*, *"skip"*, *"louder"*, *"what's playing?"*
- *"loop this"*, *"play Weightless on repeat"*
- Playlists — *"what playlists do I have"*, *"play my Focus playlist"*, *"add this to Favourites"*
- Your library — *"play my most-played song"*, *"my top songs/artists"*, *"my liked songs"*, *"what did I play recently"*

### 🌐 Browser & web (Chrome)
- *"open instagram"*, *"play some lofi on youtube"*, *"open a new MrBeast video"*, *"show me reels"*
- Live info via **Tavily** web search — *"latest news on X"*, *"current price of bitcoin"*, *"weather tomorrow"*
- **Browser agent** (logged-in, multi-step workflows) — *"download my OS notes from VTOP"*,
  *"check my AWS balance"*. Drives your **real Chrome profile** via
  [browser-use](https://github.com/browser-use/browser-use) in an isolated venv.
  One-time setup: `bash scripts/setup_browser_agent.sh`.
- **Assignment workflow** — *"download the latest assignment and explain it"* →
  *"finish it as a notebook"* → *"open it"*: downloads, explains aloud, has an AI
  complete it, assembles the file, and opens it.

### 🖥️ Screen vision & on-screen document help
- *"what's on my screen?"*, *"explain this error"*, *"read this for me"* → screenshots
  and explains what's displayed (`show_in_ui` renders it on the HUD).
- *"explain this topic, I don't understand it"* → knows **which document is live on
  screen**, works out the **page/slide** you're on, and reads the **surrounding
  pages of that subtopic** from your indexed files (not just the visible screenshot)
  to explain it properly.

### 📄 Your documents (local RAG)
JARVIS indexes your PDFs / Word / PowerPoint / notes into a **local, self-updating
vector store** (new downloads added, changed files re-embedded, deleted files
dropped — without re-embedding everything). Ask about them by **description — no
exact filename needed**:
- *"what does my machine-learning assignment say about backprop?"*, *"explain page 4 of my calculus notes"*
- *"summarise my thermodynamics notes"*, *"what's slide 5 about"*
- *"open my OS notes"* (opens the right file), *"where's my budget spreadsheet / when did I download it"*
- If several files match, JARVIS reads back the top few and asks which one.

Embeddings are **local** (Ollama `nomic-embed-text`); answering runs on Cerebras.
Re-scan happens automatically; force a full rebuild with `python scripts/reindex_rag.py`.

### 🗂️ Sandboxed file organiser
Read / copy / move / open — and tidy folders into category subfolders. **Never
deletes anything** (duplicates are moved aside), and is **confined to your home
folder** (it can't move files outside the sandbox).
- *"organise my downloads"*, *"move X to Y"*, *"open my recent downloads"*, *"what's in my Documents"*

### 📅 Calendar, to-dos, reminders & alarms
Time-aware: computes absolute times from natural language.
- *"remind me to submit the assignment at 9pm"*, *"remind me in an hour"*, *"add milk to my to-do list"*
- A due reminder **rings a looping alarm**, pops up on the HUD (STOP / MARK DONE), and speaks. *"Jarvis, stop the alarm"* / *"I'm done with X"*.

### 🎯 Focus assist mode
- *"focus mode"*, *"start a pomodoro"*, *"deep work session"* → **closes every open
  distraction** (Instagram, YouTube, Netflix, TikTok, Reddit, X, …), starts a timer,
  and **keeps closing anything distracting you reopen** — announcing it — until you
  say *"end focus mode"*. If you ask it to open a blocked site, it refuses.
- Techniques: **pomodoro** (25/5), **classic** (50/10), **52-17**, **90-20** (ultradian), **flowtime**.
- Shown live on the HUD with a countdown, phase (work/break), and sprint count.

### 🟦 The HUD dashboard
A hidden Iron-Man-style interface that **opens automatically on launch** (⌄ HIDE
button to dismiss; say *"show the dashboard"* to bring it back). Shows your
profile/memories, past conversations, agenda & to-dos, a **live Q&A feed** of
everything you ask and every answer, **rendered explanations** (e.g. a screen
analysis with the captured image), the **focus timer**, and **alarm pop-ups**.
Served locally only.

### 🧠 Persistent memory (local)
Learns what you like from what you do, from *"remember that I…"*, and from facts
mined out of conversation in the background — recalled each turn to personalise
replies. *"play me something I'd enjoy"*, *"what do you know about me?"*, *"forget that…"*.
Lives in `~/.jarvis/memory/` (local JSON + Ollama embeddings).

---

## Two pipelines (`JARVIS_MODE`)

| | `1` CLOUD (default) | `0` LOCAL |
|---|---|---|
| **STT** | Deepgram `nova-3` (streaming) | Whisper mlx (`small.en`) |
| **Brain** | Cerebras `gpt-oss-120b` → OpenAI | Ollama `qwen2.5:7b-instruct` |
| **Vision** | Cerebras `gemma-4-31b` → gpt-4o | (uses the cloud vision key) |
| **TTS** | Deepgram Aura-2 (`aura-2-draco-en`) | Kokoro-82M |

**CLOUD** is fast and hosted (audio → Deepgram, text/tools → Cerebras). **LOCAL**
keeps the voice loop fully on-device (Whisper + Kokoro + Ollama); the only egress
is the optional Spotify catalog search (a song title). Note that document RAG and
long-term memory always use **local** Ollama embeddings in either mode.

---

## Quick start

**Clickable launcher (no terminal):** double-click **`scripts/Start JARVIS.command`**
in Finder, or build a proper Mac app with `bash scripts/make_app.sh` (creates
`~/Applications/JARVIS.app` — drag it to your Dock). First launch: right-click ▸
**Open** to get past Gatekeeper.

**Terminal:**
```bash
source .venv/bin/activate
python -m jarvis.agent console
```
Say **"Hey Jarvis,"** then your request (follow-ups within ~20s skip the wake word):
*"Play Bohemian Rhapsody."* · *"What's on my screen?"* · *"Focus mode."* ·
*"Summarise my OS notes."* · *"Remind me to submit at 9pm."* · *"What's on my calendar?"*

---

## Setup & API keys

### 1. Python environment (Python 3.11+; 3.13 tested)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in keys below
```

### 2. Cloud keys — needed for the default pipeline
| Service | Used for | Get a key |
|---|---|---|
| **Cerebras** | the brain + RAG answering + screen vision (fast) | [cloud.cerebras.ai](https://cloud.cerebras.ai) · [docs](https://inference-docs.cerebras.ai) |
| **Deepgram** | speech-to-text + text-to-speech | [console.deepgram.com/signup](https://console.deepgram.com/signup) |

Put them in `.env`:
```ini
CEREBRAS_API_KEY=csk-...
DEEPGRAM_API_KEY=...
```
That's the minimum for a talking assistant in cloud mode.

### 3. Ollama — for embeddings (RAG + memory), and for LOCAL mode
Document RAG and long-term memory use a **local** embedding model, so install
Ollama even in cloud mode:
```bash
brew install ollama
ollama serve                        # or: brew services start ollama
ollama pull nomic-embed-text        # embeddings for RAG + memory
# LOCAL mode (JARVIS_MODE=0) also needs a chat model:
ollama pull qwen2.5:7b-instruct
```
Ollama: [ollama.com](https://ollama.com). The clickable launcher starts `ollama serve` for you if it's installed.

### 4. Optional keys — unlock more features
| Service | Unlocks | Get a key |
|---|---|---|
| **OpenAI** | fallback brain/vision; the browser agent & assignment workflow | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Tavily** | live web search (news, prices, weather) | [app.tavily.com](https://app.tavily.com) (free tier) |
| **Spotify** | music search, playlists & library | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) |

```ini
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
```

**Spotify playlists & library** need a one-time user login (search alone only needs
the client id/secret). Add `http://127.0.0.1:8080/callback` to your app's Redirect
URIs, then:
```bash
python -m jarvis.spotify_auth
```

### 5. LOCAL-mode voice model (only if you set `JARVIS_MODE=0`)
Download the Kokoro TTS files once (~350 MB):
```bash
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 6. Browser agent (optional — logged-in web workflows)
Runs in a separate venv because its deps conflict with the main stack:
```bash
bash scripts/setup_browser_agent.sh
# optional local captcha solver: ollama pull qwen2.5vl:7b
```
Reasoning uses a frontier model via `OPENAI_API_KEY`. ⚠️ It drives your **real
logged-in Chrome**, so browsed page content goes to OpenAI and a malicious page
could prompt-inject it — it runs only when you ask, with step/time caps.

---

## macOS permissions

Grant these to your terminal (or `JARVIS.app`) under **System Settings ▸ Privacy &
Security** — most are prompted on first use:

- **Microphone** — to hear you.
- **Screen Recording** — for *"explain my screen"* and on-screen document help.
- **Automation** — to control Google Chrome & System Events (browser control,
  media control, **focus mode** closing tabs).
- **Accessibility** — lets JARVIS read the *focused window's title* so it knows
  which open document you mean (falls back gracefully without it).

---

## Configuration reference

All settings live in **`.env`** (copied from **`.env.example`**, which documents
every option). The essentials:

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_MODE` | `1` | `1` cloud, `0` local |
| `JARVIS_ORCHESTRATOR` | `openai` | full-featured single-agent brain (or `langgraph` / `native`) |
| `JARVIS_AGENT_PROVIDER` | `cerebras` | `cerebras` (fast) or `openai`; falls back to OpenAI if no Cerebras key |
| `JARVIS_WAKE` / `JARVIS_WAKE_WORDS` | `1` / `jarvis,…` | wake word on/off + trigger words |
| `JARVIS_UI` | `1` | HUD dashboard (auto-opens on launch) |
| `JARVIS_RAG` / `JARVIS_RAG_DIRS` | `1` / Downloads,Documents,Desktop | document indexing + which folders |
| `JARVIS_FILES` / `JARVIS_FILES_SANDBOX` | `1` / `~` | file organiser + sandbox root |
| `JARVIS_SCHEDULER` | `1` | calendar / to-dos / reminders / alarms |
| `JARVIS_FOCUS` / `JARVIS_FOCUS_TECHNIQUE` | `1` / `pomodoro` | focus mode + default technique |
| `JARVIS_BROWSER_AGENT` | `1` | logged-in browser workflows |

Model tuning (`CEREBRAS_MODEL`, `JARVIS_CEREBRAS_VISION_MODEL`, `WHISPER_MODEL`,
`OLLAMA_MODEL`, endpointing delays, RAG scan cadence, focus blocklist, …) is all in
`.env.example` with inline notes.

---

## Architecture

```
                 ┌─────────── LiveKit voice pipeline ───────────┐
mic ─► VAD ─► STT ─► [wake gate] ─► BRAIN (agent + tools + memory) ─► TTS ─► speaker
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
     Spotify   Browser/Web   Screen vision   Documents(RAG)   Files
     Calendar/Reminders     Focus mode      HUD dashboard     Memory
```

- **Wake word with smart follow-up:** say *"jarvis"* / *"hey jarvis"* to activate
  (the word is stripped). After a **clarifying question** JARVIS stays awake so you
  answer directly; after a normal answer it sleeps until the next wake word.
  `JARVIS_WAKE=0` replies to everything.
- **One agent, all tools:** a single agent holds every capability. Device actions
  resolve in one fast round-trip; informational tools keep the loop open so
  multi-step requests complete and are synthesised into one answer. `gpt-oss` runs
  at low reasoning effort for speed (~0.5–0.8s/step on Cerebras).
- **Custom LiveKit LLM adapter:** the whole agent framework is plugged in as the
  LLM node of the STT→LLM→TTS pipeline; STT/TTS are unchanged.

---

## Privacy & safety

- **File organiser can never delete** — only read / copy / move / open — and is
  **sandboxed to your home folder** (enforced structurally; there are no
  delete/remove operations anywhere in the file layer).
- **Local by default where it matters:** document embeddings and long-term memory
  are computed locally (Ollama) and stored as local JSON under `~/.jarvis/`.
- **LOCAL mode** (`JARVIS_MODE=0`) keeps the entire voice loop on-device.
- **`.env` holds secrets and is gitignored** — never commit it.
- **Cloud egress, made explicit:** in cloud mode, audio goes to Deepgram and your
  text/tool calls + screenshots go to Cerebras/OpenAI. The browser agent sends
  browsed page content to OpenAI and acts on your logged-in sessions.

---

## Handy scripts
- `scripts/Start JARVIS.command` — double-click launcher.
- `scripts/make_app.sh` — build `JARVIS.app`.
- `scripts/setup_browser_agent.sh` — install the isolated browser-agent venv.
- `scripts/reindex_rag.py` — force a full document re-index.
- `python -m jarvis.spotify_auth` — authorise Spotify playlists/library.
