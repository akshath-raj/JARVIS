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
    cloud: bool = _truthy(os.getenv("JARVIS_MODE", "0"))
    # In cloud mode, run local device-action agents (music/calendar/files) on the
    # LOCAL LLM for reliable, private tool execution. Answering stays cloud.
    hybrid_local_actions: bool = _truthy(os.getenv("JARVIS_HYBRID_LOCAL_ACTIONS", "1"))

    # ── Local pipeline (per-task models; smaller = faster) ─────────────
    # Chat / router / general reasoning.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    # Device-action agents — a compact NON-THINKING model for snappy tool calls
    # (thinking models like qwen3 burn seconds "reasoning" before every call).
    tool_model: str = os.getenv("JARVIS_TOOL_MODEL", "qwen2.5:3b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # STT backend: "mlx" (Metal, fastest on Apple Silicon) or "faster" (CPU).
    whisper_backend: str = os.getenv("WHISPER_BACKEND", "mlx")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base.en")

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

    # ── Activation ─────────────────────────────────────────────────────
    # Wake word is OFF by default: JARVIS listens continuously and a relevance
    # gate decides whether each utterance is actually addressed to it.
    wake_enabled: bool = _truthy(os.getenv("JARVIS_WAKE", "0"))
    wake_model: str = os.getenv("JARVIS_WAKE_MODEL", "hey_jarvis")
    wake_threshold: float = float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5"))
    wake_active_seconds: float = float(os.getenv("JARVIS_WAKE_WINDOW", "12"))

    # Context-aware relevance gate (directed-speech vs background noise).
    relevance_enabled: bool = _truthy(os.getenv("JARVIS_RELEVANCE", "1"))
    relevance_model: str = os.getenv("JARVIS_RELEVANCE_MODEL", "qwen2.5:1.5b-instruct")

    # ── Spotify ────────────────────────────────────────────────────────
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    spotify_search_mode: str = os.getenv("SPOTIFY_SEARCH_MODE", "auto")
    # For playlist features (user OAuth). Add this exact URI to your Spotify app.
    spotify_redirect_uri: str = os.getenv(
        "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080/callback"
    )


config = Config()
