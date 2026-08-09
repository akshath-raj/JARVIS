"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # LLM (Ollama, OpenAI-compatible)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # STT (faster-whisper)
    whisper_model: str = os.getenv("WHISPER_MODEL", "base.en")

    # TTS (Kokoro)
    kokoro_voice: str = os.getenv("KOKORO_VOICE", "bm_george")
    kokoro_model_path: str = os.getenv("KOKORO_MODEL_PATH", "models/kokoro-v1.0.onnx")
    kokoro_voices_path: str = os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin")

    # Spotify
    spotify_client_id: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    spotify_client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    # How to resolve a song name: "app" (UI only), "web" (Web API only),
    # or "auto" (try the local app first, fall back to the Web API).
    spotify_search_mode: str = os.getenv("SPOTIFY_SEARCH_MODE", "auto")


config = Config()
