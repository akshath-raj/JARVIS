"""Central configuration, loaded from environment / .env.

JARVIS runs in one of two modes, selected by JARVIS_MODE:
  0 = LOCAL  (default): faster/mlx Whisper + Ollama (qwen3) + Kokoro. No cloud.
  1 = CLOUD          : Deepgram STT + (Cerebras|OpenAI) LLM + Cartesia TTS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Always load the project's own .env (parent of this package), regardless of the
# current working directory, so keys resolve the same way from anywhere.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on", "cloud")


@dataclass(frozen=True)
class Config:
    # ── Mode ───────────────────────────────────────────────────────────
    # Default is CLOUD voice now (Deepgram nova-3 STT + Deepgram Aura-2 TTS);
    # set JARVIS_MODE=0 for the fully-local Whisper + Kokoro stack.
    cloud: bool = _truthy(os.getenv("JARVIS_MODE", "1"))
    # In cloud mode, run local device-action agents (music/calendar/files) on the
    # LOCAL LLM for reliable, private tool execution. Answering stays cloud.
    hybrid_local_actions: bool = _truthy(os.getenv("JARVIS_HYBRID_LOCAL_ACTIONS", "1"))

    # ── Local pipeline (per-task models; smaller = faster) ─────────────
    # The single agent's LLM (answers + Spotify tool-calls). qwen2.5:7b-instruct is
    # non-thinking (fast, no 10-40s qwen3 "thinking") and reliable at tool calls on
    # arbitrary songs. Drop to :3b for speed (less reliable) or up for more quality.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    # Device-action agents — non-thinking, reliable tool-caller on ANY song/artist
    # (7B: 8/8 in testing; 3B misfired — refused some titles, hallucinated loop).
    # ~2s/call. Drop to qwen2.5:3b-instruct for speed if you accept less accuracy.
    tool_model: str = os.getenv("JARVIS_TOOL_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # STT backend: "mlx" (Metal, fastest on Apple Silicon) or "faster" (CPU).
    whisper_backend: str = os.getenv("WHISPER_BACKEND", "mlx")
    # small.en is markedly more accurate than base.en on song/artist names, still
    # fast on Metal. Use medium.en for max accuracy, tiny.en for max speed.
    whisper_model: str = os.getenv("WHISPER_MODEL", "small.en")

    kokoro_voice: str = os.getenv("KOKORO_VOICE", "bm_george")
    kokoro_model_path: str = os.getenv("KOKORO_MODEL_PATH", "models/kokoro-v1.0.onnx")
    kokoro_voices_path: str = os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin")

    # ── Cloud pipeline (mirrors the reference "live JARVIS" stack) ──────
    cerebras_api_key: str = os.getenv("CEREBRAS_API_KEY", "")
    cerebras_model: str = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Deepgram handles both STT (nova-3) and TTS (Aura-2) in cloud mode.
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")
    deepgram_tts_model: str = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-draco-en")

    # ── Turn detection ─────────────────────────────────────────────────
    # Use the LiveKit MultilingualModel turn detector (better than VAD-only).
    use_turn_detector: bool = _truthy(os.getenv("JARVIS_TURN_DETECTOR", "1"))
    # Endpointing: JARVIS only assumes you've finished once your speech gap exceeds
    # `endpoint_delay` seconds (400ms default) — so short breaths don't cut you off.
    # When your sentence still sounds unfinished, the turn detector keeps waiting up
    # to `endpoint_max_delay`, giving you room to think mid-thought.
    endpoint_delay: float = float(os.getenv("JARVIS_ENDPOINT_DELAY", "0.4"))
    endpoint_max_delay: float = float(os.getenv("JARVIS_ENDPOINT_MAX_DELAY", "6.0"))

    # ── Activation (wake word + smart follow-up) ───────────────────────
    # Say "jarvis" or "hey jarvis" to activate. After JARVIS asks a clarifying
    # question, the next turn is answered WITHOUT the wake word for a short window;
    # after a normal answer it goes back to sleep.
    wake_enabled: bool = _truthy(os.getenv("JARVIS_WAKE", "1"))
    # Include common Whisper mishearings of "jarvis" (jairus/jarvus/jervis/javis) so
    # the wake gate still fires when the STT slightly mistranscribes the name.
    wake_words: tuple[str, ...] = tuple(
        w.strip()
        for w in os.getenv(
            "JARVIS_WAKE_WORDS", "jarvis,jairus,jarvus,jervis,javis"
        ).split(",")
        if w.strip()
    )
    wake_followup_seconds: float = float(os.getenv("JARVIS_WAKE_FOLLOWUP", "5"))
    # After a normal (non-question) answer, optionally stay awake briefly so
    # back-to-back commands work without repeating the wake word. OFF by default:
    # when music/video plays through SPEAKERS the mic transcribes it as fake user
    # turns while awake, which makes JARVIS re-run tools in a loop. Safe to enable
    # (e.g. 8) only with headphones/good echo cancellation.
    wake_continuation_seconds: float = float(os.getenv("JARVIS_WAKE_CONTINUATION", "0"))

    # ── Orchestrator / brain ───────────────────────────────────────────
    # "langgraph" (default): LangGraph react-agent brain + persistent memory.
    # "openai": OpenAI Agents SDK — a triage agent that HANDS OFF to specialist
    #           agents (Music / Browser / Screen), all running on a frontier model.
    #           This is the "cloud brain" with full feature parity + screen vision.
    # "native": the previous hand-rolled LiveKit supervisor (fallback / rollback).
    orchestrator: str = os.getenv("JARVIS_ORCHESTRATOR", "openai")
    # Which provider backs the agent brain: "cerebras" (default — OpenAI-compatible,
    # ~2000+ tok/s, much lower latency) or "openai". When "cerebras", the brain,
    # RAG answering AND screen vision all run on Cerebras (text on cerebras_model,
    # images on cerebras_vision_model) for uniformly fast responses. Falls back to
    # OpenAI (cloud_agent_model / vision_model) if the Cerebras key is missing.
    agent_provider: str = os.getenv("JARVIS_AGENT_PROVIDER", "cerebras")
    cerebras_base_url: str = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    # Cerebras multimodal model for images (screenshots/docs/charts) — Gemma 4 31B,
    # ~1800 tok/s, OpenAI-compatible image_url inputs. Used for screen vision + RAG
    # answering when agent_provider="cerebras".
    cerebras_vision_model: str = os.getenv("JARVIS_CEREBRAS_VISION_MODEL", "gemma-4-31b")
    # Frontier model the OpenAI Agents SDK brain runs on when agent_provider="openai".
    cloud_agent_model: str = os.getenv("JARVIS_CLOUD_AGENT_MODEL", "gpt-5.4-mini")
    # OpenAI vision model used when agent_provider="openai" (or Cerebras key absent).
    vision_model: str = os.getenv("JARVIS_VISION_MODEL", "gpt-4o")

    # ── HUD dashboard UI (hidden; appears on voice command) ─────────────
    # A local Iron-Man-style dashboard JARVIS reveals when you ask. It shows your
    # profile/memories/past conversations and renders explanations (e.g. a screen
    # analysis) on screen. Served locally; nothing leaves the machine.
    ui_enabled: bool = _truthy(os.getenv("JARVIS_UI", "1"))
    ui_port: int = int(os.getenv("JARVIS_UI_PORT", "8137"))
    ui_user: str = os.getenv("JARVIS_UI_USER", "Akshath")

    # ── Document RAG (local embeddings + self-updating vector store) ───
    # JARVIS indexes your documents (PDF/DOCX/PPTX/TXT/MD) and answers questions
    # about them. Embeddings are LOCAL by default (Ollama nomic-embed-text) so the
    # corpus never leaves the machine; only retrieved snippets go to the answer LLM.
    rag_enabled: bool = _truthy(os.getenv("JARVIS_RAG", "1"))
    rag_embed_backend: str = os.getenv("JARVIS_RAG_EMBED", "ollama")  # ollama | openai
    rag_embed_model: str = os.getenv("JARVIS_RAG_EMBED_MODEL", "nomic-embed-text")
    rag_index_dir: str = os.path.expanduser(os.getenv("JARVIS_RAG_INDEX_DIR", "~/.jarvis/rag"))
    # Folders scanned for documents to index (comma-separated). All under the home.
    # Default is just Downloads (where files land); add Documents/Desktop if you want
    # them indexed too — but a big/cluttered tree means a heavy first embed.
    rag_dirs: tuple[str, ...] = tuple(
        d.strip() for d in os.getenv("JARVIS_RAG_DIRS", "~/Downloads").split(",")
        if d.strip()
    )
    # Auto-ingest: rescan the folders this often (seconds). 0 disables it. The first
    # scan is delayed so it never competes with startup, and each cycle indexes at
    # most `rag_max_per_scan` new files so it spreads the work out (stays responsive).
    rag_scan_seconds: int = int(os.getenv("JARVIS_RAG_SCAN_SECONDS", "300"))
    rag_scan_start_delay: int = int(os.getenv("JARVIS_RAG_SCAN_START_DELAY", "120"))
    rag_max_per_scan: int = int(os.getenv("JARVIS_RAG_MAX_PER_SCAN", "12"))

    # ── Sandboxed file organiser (read / copy / move / open — never delete) ──
    files_enabled: bool = _truthy(os.getenv("JARVIS_FILES", "1"))
    # Everything JARVIS touches must stay under this root (default: your home).
    files_sandbox: str = os.path.expanduser(os.getenv("JARVIS_FILES_SANDBOX", "~"))

    # ── Calendar / to-dos / reminders + alarms ─────────────────────────
    # Time-aware planning: JARVIS can set reminders that ring an alarm, keep a
    # to-do list, and manage a calendar/agenda (shown on the HUD).
    scheduler_enabled: bool = _truthy(os.getenv("JARVIS_SCHEDULER", "1"))
    # macOS system sound looped as the alarm (empty = default Sosumi).
    alarm_sound: str = os.getenv("JARVIS_ALARM_SOUND", "")

    # ── Focus assist mode ───────────────────────────────────────────────
    # "focus mode on" closes open distractions (Instagram/YouTube/Netflix/…),
    # runs a Pomodoro (or another technique), and keeps closing any distraction
    # you reopen — until "end focus mode".
    focus_enabled: bool = _truthy(os.getenv("JARVIS_FOCUS", "1"))
    focus_default_technique: str = os.getenv("JARVIS_FOCUS_TECHNIQUE", "pomodoro")
    focus_poll_seconds: int = int(os.getenv("JARVIS_FOCUS_POLL", "3"))  # distraction-scan cadence
    # extra comma-separated distraction domains / macOS app names to block (added
    # to the built-in defaults). Apps are only quit if the user lists them here.
    focus_block_sites: str = os.getenv("JARVIS_FOCUS_BLOCK_SITES", "")
    focus_block_apps: str = os.getenv("JARVIS_FOCUS_BLOCK_APPS", "")
    # Local embedding model (kept for optional future use; memory no longer indexes).
    embed_model: str = os.getenv("JARVIS_EMBED_MODEL", "nomic-embed-text")
    embed_dims: int = int(os.getenv("JARVIS_EMBED_DIMS", "768"))  # nomic-embed-text = 768
    # Where the distilled profile (about_user.md) + transient log (activity.jsonl)
    # persist. ~ is expanded.
    memory_dir: str = os.path.expanduser(os.getenv("JARVIS_MEMORY_DIR", "~/.jarvis/memory"))
    # Small, fast model used off the voice path to distill the activity/conversation
    # log into the durable user profile.
    extract_model: str = os.getenv("JARVIS_EXTRACT_MODEL", "qwen2.5:3b-instruct")
    # Distill the transient log into about_user.md after every N logged entries,
    # then clear the log (so raw prompts/activity are never kept permanently).
    summarize_every: int = int(
        os.getenv("JARVIS_SUMMARIZE_EVERY", os.getenv("JARVIS_EXTRACT_EVERY", "15"))
    )

    # ── Browser / Web (Chrome + Tavily) ────────────────────────────────
    # Which browser app to drive (macOS app name).
    browser_app: str = os.getenv("JARVIS_BROWSER_APP", "Google Chrome")
    # Tavily powers live web search / recent news. Get a key at tavily.com.
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # ── Browser AGENT (browser-use, isolated subprocess) ───────────────
    # Multi-step, logged-in web workflows (download VTOP materials, check AWS
    # balance). Runs in .venv-browser out-of-process; see scripts/setup_browser_agent.sh.
    browser_agent_enabled: bool = _truthy(os.getenv("JARVIS_BROWSER_AGENT", "1"))
    browser_venv: str = os.getenv("JARVIS_BROWSER_VENV", ".venv-browser")
    # Frontier model does the reasoning (local models can't drive a browser
    # reliably — verified). Uses OPENAI_API_KEY.
    browser_frontier_model: str = os.getenv("JARVIS_BROWSER_FRONTIER_MODEL", "gpt-4.1-mini")
    # Experimental local-first path, OFF by default (it hallucinates + false-successes).
    browser_use_local: bool = _truthy(os.getenv("JARVIS_BROWSER_USE_LOCAL", "0"))
    browser_local_model: str = os.getenv("JARVIS_BROWSER_LOCAL_MODEL", "qwen2.5:7b-instruct")
    # Real Chrome profile (saved logins/passwords). Chrome must be CLOSED when a task runs.
    browser_chrome_path: str = os.getenv(
        "JARVIS_BROWSER_CHROME_PATH",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    browser_user_data_dir: str = os.path.expanduser(
        os.getenv("JARVIS_BROWSER_USER_DATA_DIR", "~/Library/Application Support/Google/Chrome")
    )
    browser_profile_dir: str = os.getenv("JARVIS_BROWSER_PROFILE", "Default")
    # A Chrome profile dir can only be used by ONE Chrome process at a time. To run
    # headless in the background WHILE your Chrome is open, JARVIS clones the profile
    # (cookies/logins, minus caches) to its own dir and drives that copy.
    #   auto (default) = clone only when Chrome is running; else use the real profile.
    #   always = always clone (headless); never = require Chrome closed (no clone).
    browser_clone_profile: str = os.getenv("JARVIS_BROWSER_CLONE_PROFILE", "auto")
    browser_clone_dir: str = os.path.expanduser(
        os.getenv("JARVIS_BROWSER_CLONE_DIR", "~/.jarvis/chrome-clone")
    )
    browser_downloads: str = os.path.expanduser(os.getenv("JARVIS_BROWSER_DOWNLOADS", "~/Downloads"))
    browser_max_steps: int = int(os.getenv("JARVIS_BROWSER_MAX_STEPS", "40"))
    browser_local_max_steps: int = int(os.getenv("JARVIS_BROWSER_LOCAL_MAX_STEPS", "12"))
    browser_timeout: int = int(os.getenv("JARVIS_BROWSER_TIMEOUT", "240"))
    browser_headless: bool = _truthy(os.getenv("JARVIS_BROWSER_HEADLESS", "0"))
    # Auto-solve captchas that block a task, using a LOCAL open-source vision model
    # (Qwen2.5-VL via Ollama) — no third-party service. Needs: ollama pull qwen2.5vl:7b.
    captcha_enabled: bool = _truthy(os.getenv("JARVIS_BROWSER_CAPTCHA", "1"))
    captcha_model: str = os.getenv("JARVIS_CAPTCHA_MODEL", "qwen2.5vl:7b")

    # ── Documents / assignments ────────────────────────────────────────
    # Preferred AI apps for "do the assignment", in order. The first AVAILABLE one
    # wins: claude_desktop is only available if Claude.app is installed AND macOS
    # Accessibility is granted to this process, so by default it falls through to
    # the browser (claude.ai). An "openai" API backend can be added later.
    ai_app_order: tuple[str, ...] = tuple(
        w.strip() for w in os.getenv("JARVIS_AI_APP_ORDER", "claude_desktop,browser").split(",") if w.strip()
    )
    # Model that reads & explains a downloaded document (local by default).
    explain_model: str = os.getenv("JARVIS_EXPLAIN_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    # Where assembled answer files are written.
    answers_dir: str = os.path.expanduser(
        os.getenv("JARVIS_ANSWERS_DIR", os.getenv("JARVIS_BROWSER_DOWNLOADS", "~/Downloads"))
    )

    # ── Spotify ────────────────────────────────────────────────────────
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    spotify_search_mode: str = os.getenv("SPOTIFY_SEARCH_MODE", "auto")
    # For playlist features (user OAuth). Add this exact URI to your Spotify app.
    spotify_redirect_uri: str = os.getenv(
        "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080/callback"
    )


config = Config()
