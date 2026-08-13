# JARVIS — a voice AI assistant for macOS

![Platform: macOS](https://img.shields.io/badge/platform-macOS-black?logo=apple)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Local-first](https://img.shields.io/badge/local--first-Ollama%20%2B%20Whisper%20%2B%20Kokoro-orange)

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

> **Platform:** macOS only (Apple Silicon recommended). See
> [Platform support](#platform-support) for why, and Windows/Linux notes.

---

## Contents
- [Feature overview](#feature-overview)
- [Two pipelines (`JARVIS_MODE`)](#two-pipelines-jarvis_mode)
- [Install & run — step by step](#install--run--step-by-step)
- [Optional features & their setup](#optional-features--their-setup)
- [Platform support](#platform-support)
- [Configuration reference](#configuration-reference)
- [Architecture](#architecture)
- [Privacy & safety](#privacy--safety)
- [Troubleshooting](#troubleshooting)

---

## Feature overview

Everything below works with the default `openai` brain (a single agent on
Cerebras/OpenAI). Say the wake word, then your request.

### 🎵 Music (Spotify)
Playback runs **in the background** (launched hidden, driven by AppleScript) so it
never steals focus. Needs the Spotify desktop app installed + logged in.
- *"play Bohemian Rhapsody by Queen"*, *"pause"*, *"skip"*, *"louder"*, *"what's playing?"*
- *"loop this"*, *"play Weightless on repeat"*
- Playlists — *"what playlists do I have"*, *"play my Focus playlist"*, *"add this to Favourites"*
- Your library — *"play my most-played song"*, *"my top songs/artists"*, *"my liked songs"*

### 🌐 Browser & web (Chrome)
- *"open instagram"*, *"play some lofi on youtube"*, *"open a new MrBeast video"*, *"show me reels"*
- Live info via **Tavily** web search — *"latest news on X"*, *"price of bitcoin"*, *"weather tomorrow"*
- **Browser agent** (logged-in, multi-step) — *"download my OS notes from VTOP"*, *"check my AWS balance"*
- **Assignment workflow** — *"download the latest assignment and explain it"* → *"finish it as a notebook"* → *"open it"*

### 🖥️ Screen vision & on-screen document help
- *"what's on my screen?"*, *"explain this error"*, *"read this for me"* → screenshots and explains.
- *"explain this topic, I don't understand it"* → knows **which document is live on
  screen**, the **page/slide** you're on, and reads the **surrounding pages of that
  subtopic** from your indexed files to explain it properly.

### 📄 Your documents (local RAG)
JARVIS indexes your PDFs / Word / PowerPoint / notes into a **local, self-updating**
vector store. Ask by **description — no exact filename needed**:
- *"what does my ML assignment say about backprop?"*, *"explain page 4 of my calculus notes"*
- *"summarise my thermodynamics notes"*, *"open my OS notes"*, *"where's my budget spreadsheet?"*

### 🗂️ Sandboxed file organiser
Read / copy / move / open / tidy folders. **Never deletes** and is **confined to
your home folder**. *"organise my downloads"*, *"move X to Y"*, *"open my recent downloads"*.

### 📅 Calendar, to-dos, reminders & alarms
*"remind me to submit at 9pm"*, *"remind me in an hour"*, *"add milk to my list"*. A
due reminder **rings an alarm**, pops up on the HUD, and speaks; *"stop the alarm"* / *"I'm done with X"*.

### 🎯 Focus assist mode
*"focus mode"*, *"start a pomodoro"* → **closes every open distraction** (Instagram,
YouTube, Netflix, TikTok, Reddit, X…), starts a timer, and **keeps closing anything
you reopen** until *"end focus mode"*. Techniques: **pomodoro / classic / 52-17 /
90-20 / flowtime**. Shown live on the HUD with a countdown.

### 🟦 The HUD dashboard
A hidden Iron-Man-style interface that **opens automatically on launch** (⌄ HIDE to
dismiss; say *"show the dashboard"* to bring it back). Shows your profile/memories,
past conversations, agenda, a **live Q&A feed**, **rendered explanations**, the
**focus timer**, and **alarm pop-ups**. Local only.

### 🧠 Persistent memory (local)
Learns what you like and recalls it to personalise replies. *"remember that I…"*,
*"what do you know about me?"*, *"forget that…"*. Stored in `~/.jarvis/memory/`.

---

## Two pipelines (`JARVIS_MODE`)

| | `1` CLOUD (default) | `0` LOCAL |
|---|---|---|
| **STT** | Deepgram `nova-3` (streaming) | Whisper mlx (`small.en`) |
| **Brain** | Cerebras `gpt-oss-120b` → OpenAI | Ollama `qwen2.5:7b-instruct` |
| **Vision** | Cerebras `gemma-4-31b` → gpt-4o | (uses the cloud vision key) |
| **TTS** | Deepgram Aura-2 (`aura-2-draco-en`) | Kokoro-82M |

**CLOUD** is fast and hosted. **LOCAL** keeps the voice loop fully on-device.
Document RAG and long-term memory always use **local** Ollama embeddings in either mode.

---

## Install & run — step by step

> **Follow these in order on a Mac.** ~15 minutes. **Steps 1–8 give you a talking
> assistant** with document Q&A, focus mode, and the HUD. Everything else is in
> [Optional features](#optional-features--their-setup). Every command is copy-paste.

### Step 1 — Install Homebrew (skip if you already have `brew`)
Homebrew is the macOS package manager. Paste this into **Terminal** (⌘-Space → "Terminal"):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
When it finishes, run the two `echo`/`eval` lines it prints (to add `brew` to your PATH), or just close and reopen Terminal.

### Step 2 — Install the system tools
```bash
brew install python git ollama ffmpeg
```

### Step 3 — Get the code
```bash
git clone https://github.com/akshath-raj/JARVIS.git
cd JARVIS
```

### Step 4 — Create the Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> You'll run JARVIS from this folder with `.venv` active. To reactivate later:
> `cd JARVIS && source .venv/bin/activate`.

### Step 5 — Start Ollama and pull the embedding model
Document search and memory always run locally through Ollama, so this is required
even in cloud mode:
```bash
brew services start ollama      # runs Ollama in the background, and on login
ollama pull nomic-embed-text
```

### Step 6 — Get your two free API keys
You need two keys for the default (cloud) setup:

| Key | What it powers | Where to get it |
|---|---|---|
| **Cerebras** | the brain, document answers, screen vision | 1. Sign up at **[cloud.cerebras.ai](https://cloud.cerebras.ai)** → 2. **API Keys** → **Create**. It starts with `csk-`. |
| **Deepgram** | speech-to-text + text-to-speech (the voice) | 1. Sign up at **[console.deepgram.com/signup](https://console.deepgram.com/signup)** (free credit) → 2. **Create API Key**. |

### Step 7 — Create your `.env` and paste the keys
```bash
cp .env.example .env
open -e .env            # opens the file in TextEdit
```
In the file that opens, find these two lines and paste your keys after the `=`:
```ini
CEREBRAS_API_KEY=csk-paste-your-cerebras-key-here
DEEPGRAM_API_KEY=paste-your-deepgram-key-here
```
**Save (⌘S) and close.** That's the minimum — everything else in `.env` already has
sensible defaults. (Your `.env` holds secrets and is git-ignored; never share it.)

### Step 8 — Run it
```bash
python -m jarvis.agent console
```
- The **HUD dashboard opens automatically** and boots with a "Welcome back" animation.
- macOS pops up a **Microphone** permission request — click **Allow**.
- Say: **"Hey Jarvis, what's the tallest mountain in the world?"**

Prefer no terminal? Double-click **`scripts/Start JARVIS.command`** in Finder (first
time: right-click ▸ **Open** ▸ **Open** to get past Gatekeeper). Or build a real app
with `bash scripts/make_app.sh` → `~/Applications/JARVIS.app` (drag it to your Dock).

### Step 9 — Grant permissions as features ask for them
Open **System Settings ▸ Privacy & Security** and add your **Terminal** (or
`JARVIS.app`) under each of these. Most are also prompted automatically the first
time you use the feature:

| Permission | Needed for |
|---|---|
| **Microphone** | hearing you (prompted on first run) |
| **Screen Recording** | *"explain my screen"* / on-screen document help |
| **Automation** | controlling Chrome & System Events — music, browser, **focus mode** |
| **Accessibility** | reading the focused window's title (which document you mean) |

> After granting a permission, **quit and reopen** the terminal/app so it takes effect.

**You're done.** Try: *"Play some lofi on YouTube."* · *"Focus mode."* ·
*"What's on my screen?"* · *"Remind me to stretch in 20 minutes."* ·
*"Summarise my notes."* (put a PDF in `~/Downloads` first).

---

## Optional features & their setup

Add any of these later — each is independent. Put keys in the same `.env`.

<details>
<summary><b>🎵 Spotify (music)</b></summary>

1. Install the **Spotify desktop app** and log in (Premium recommended).
2. Create a free app at **[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)**, copy the **Client ID** and **Client Secret** into `.env`:
   ```ini
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   ```
3. **Playlists & library** (not just search) need a one-time login. In the Spotify
   dashboard, add `http://127.0.0.1:8080/callback` to your app's **Redirect URIs**, then:
   ```bash
   python -m jarvis.spotify_auth
   ```
</details>

<details>
<summary><b>🌐 Tavily (live web search)</b></summary>

Get a free key at **[app.tavily.com](https://app.tavily.com)** and add:
```ini
TAVILY_API_KEY=tvly-...
```
Enables *"what's the latest news on…"*, *"current price of…"*, *"weather tomorrow"*.
</details>

<details>
<summary><b>🤖 OpenAI (fallback brain/vision, browser agent, assignments)</b></summary>

Get a key at **[platform.openai.com/api-keys](https://platform.openai.com/api-keys)**:
```ini
OPENAI_API_KEY=sk-...
```
Used as a fallback if Cerebras is unavailable, and required by the browser agent
and the assignment workflow (below).
</details>

<details>
<summary><b>🧭 Browser agent (logged-in web workflows)</b></summary>

Multi-step tasks on sites you're logged into (*"download my notes from VTOP"*). Runs
in its own venv because its dependencies conflict with the main stack:
```bash
bash scripts/setup_browser_agent.sh
# optional local captcha solver:
ollama pull qwen2.5vl:7b
```
Needs `OPENAI_API_KEY`. ⚠️ It drives your **real logged-in Chrome**, so browsed page
content goes to OpenAI and a malicious page could prompt-inject it — it runs only
when you ask, with step/time caps.
</details>

<details>
<summary><b>🔒 Fully local mode (no cloud, nothing leaves your Mac)</b></summary>

Set `JARVIS_MODE=0` in `.env`, then install the local chat + voice models:
```bash
ollama pull qwen2.5:7b-instruct
# Kokoro TTS voice files (~350 MB), downloaded once:
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
Local STT is mlx-whisper on Apple Silicon (auto-falls back to faster-whisper elsewhere).
</details>

---

## Platform support

**macOS only, for now.** The voice *brain* (LiveKit + Deepgram/Cerebras, or local
Whisper/Kokoro/Ollama) is cross-platform, but nearly every *action* JARVIS performs
is implemented with **macOS-native tools**:

| Capability | macOS tool used |
|---|---|
| Spotify, browser & media control, focus mode, "which window is focused" | **AppleScript** (`osascript`) |
| Screen vision (screenshots) | **`screencapture`** |
| Opening files & apps | **`open`** |
| Alarm sound | **`afplay`** |
| Fastest local STT | **mlx-whisper** (Apple Silicon / Metal) |

So on **Windows or Linux it will not work out of the box** — you'd get the voice
pipeline but none of the actions. A port is feasible but non-trivial: swap the macOS
shells for platform equivalents (e.g. `spotipy`/media keys for music, Playwright/CDP
for the browser, `mss`/`pyautogui` for screenshots, `xdg-open`/`start` for opening
files, `playsound` for alarms) and use `faster-whisper` (already the fallback)
instead of mlx-whisper. The cross-platform pieces (Ollama chat, RAG embeddings,
Tavily search) run anywhere, but there's no supported non-macOS entrypoint today.
PRs welcome.

---

## Configuration reference

All settings live in **`.env`** (copied from **`.env.example`**, which documents
every option with inline notes). The essentials:

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_MODE` | `1` | `1` cloud, `0` local |
| `JARVIS_ORCHESTRATOR` | `openai` | full-featured single-agent brain (or `langgraph` / `native`) |
| `JARVIS_AGENT_PROVIDER` | `cerebras` | `cerebras` (fast) or `openai`; falls back to OpenAI if no Cerebras key |
| `JARVIS_WAKE` / `JARVIS_WAKE_WORDS` | `1` / `jarvis,…` | wake word on/off + trigger words |
| `JARVIS_UI` | `1` | HUD dashboard (auto-opens on launch) |
| `JARVIS_RAG` / `JARVIS_RAG_DIRS` | `1` / Downloads,Documents,Desktop | document indexing + folders |
| `JARVIS_FILES` / `JARVIS_FILES_SANDBOX` | `1` / `~` | file organiser + sandbox root |
| `JARVIS_SCHEDULER` | `1` | calendar / to-dos / reminders / alarms |
| `JARVIS_FOCUS` / `JARVIS_FOCUS_TECHNIQUE` | `1` / `pomodoro` | focus mode + default technique |
| `JARVIS_BROWSER_AGENT` | `1` | logged-in browser workflows |

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
  (the word is stripped). After a **clarifying question** it stays awake; after a
  normal answer it sleeps. `JARVIS_WAKE=0` replies to everything.
- **One agent, all tools:** device actions resolve in one fast round-trip;
  informational tools keep the loop open so multi-step requests complete and are
  synthesised into one answer. `gpt-oss` runs at low reasoning effort for speed.
- **Custom LiveKit LLM adapter:** the whole agent framework is the LLM node of the
  STT→LLM→TTS pipeline; STT/TTS are unchanged.

---

## Privacy & safety

- **File organiser can never delete** — only read / copy / move / open — and is
  **sandboxed to your home folder** (enforced structurally).
- **Local where it matters:** document embeddings and long-term memory are computed
  locally (Ollama) and stored as local JSON under `~/.jarvis/`.
- **LOCAL mode** (`JARVIS_MODE=0`) keeps the entire voice loop on-device.
- **`.env` holds secrets and is git-ignored** — never commit it.
- **Cloud egress, made explicit:** in cloud mode, audio goes to Deepgram and your
  text/tool calls + screenshots go to Cerebras/OpenAI. The browser agent sends
  browsed page content to OpenAI and acts on your logged-in sessions.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute. Contributions and forks welcome.

---

## Troubleshooting

- **No response / no audio** — check your mic in System Settings ▸ Privacy ▸
  Microphone; make sure Ollama is running (`brew services start ollama`).
- **"couldn't capture the screen"** — grant **Screen Recording** to your terminal, then reopen it.
- **Focus mode / music does nothing** — grant **Automation** (Chrome + System
  Events) the first time macOS prompts; if you dismissed it, re-enable under Privacy ▸ Automation.
- **Cerebras/Deepgram errors** — re-check the keys in `.env` (no quotes, no trailing spaces).
- **Documents not found** — put files in `~/Downloads`, `~/Documents`, or `~/Desktop`;
  force a rebuild with `python scripts/reindex_rag.py`.
- **Spotify won't play** — the desktop app must be installed and logged in.

### Handy scripts
- `scripts/Start JARVIS.command` — double-click launcher.
- `scripts/make_app.sh` — build `JARVIS.app`.
- `scripts/setup_browser_agent.sh` — install the isolated browser-agent venv.
- `scripts/reindex_rag.py` — force a full document re-index.
- `python -m jarvis.spotify_auth` — authorise Spotify playlists/library.
</content>
