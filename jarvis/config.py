"""Central configuration, loaded from environment / .env.

JARVIS runs in one of two modes, selected by JARVIS_MODE:
  0 = LOCAL  (default): faster/mlx Whisper + Ollama (qwen3) + Kokoro. No cloud.
  1 = CLOUD          : Deepgram STT + (Cerebras|OpenAI) LLM + Cartesia TTS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on", "cloud")


@dataclass(frozen=True)
class Config:
    # ── Mode ───────────────────────────────────────────────────────────
    cloud: bool = _truthy(os.getenv("JARVIS_MODE", "0"))

    # ── Local pipeline ─────────────────────────────────────────────────
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
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

    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")

    cartesia_api_key: str = os.getenv("CARTESIA_API_KEY", "")
    cartesia_model: str = os.getenv("CARTESIA_MODEL", "sonic-3")
    cartesia_voice: str = os.getenv(
        "CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    )

    # ── Turn detection ─────────────────────────────────────────────────
    # Use the LiveKit MultilingualModel turn detector (better than VAD-only).
    use_turn_detector: bool = _truthy(os.getenv("JARVIS_TURN_DETECTOR", "1"))

    # ── Wake word ──────────────────────────────────────────────────────
    wake_enabled: bool = _truthy(os.getenv("JARVIS_WAKE", "1"))
    wake_model: str = os.getenv("JARVIS_WAKE_MODEL", "hey_jarvis")
    wake_threshold: float = float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5"))
    wake_active_seconds: float = float(os.getenv("JARVIS_WAKE_WINDOW", "12"))

    # ── Spotify ────────────────────────────────────────────────────────
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    spotify_search_mode: str = os.getenv("SPOTIFY_SEARCH_MODE", "auto")


config = Config()
