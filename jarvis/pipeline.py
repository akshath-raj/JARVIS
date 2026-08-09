"""Pipeline factory — swaps the STT/LLM/TTS stack between LOCAL and CLOUD.

LOCAL (JARVIS_MODE=0): Whisper (mlx/faster) + Ollama qwen3 + Kokoro. Nothing leaves
the machine.

CLOUD (JARVIS_MODE=1): mirrors the reference "live JARVIS" stack —
Deepgram nova-3 STT, Cartesia sonic-3 TTS, and an LLM that prefers Cerebras
(gpt-oss-120b) and falls back to OpenAI, whichever API key is present.
"""
from __future__ import annotations

import logging

from livekit.agents import llm as _llm
from livekit.agents import stt as _stt
from livekit.agents import tts as _tts

from jarvis.config import config

logger = logging.getLogger("jarvis.pipeline")


class ConfigError(RuntimeError):
    pass


# ── STT ────────────────────────────────────────────────────────────────
def build_stt() -> _stt.STT:
    if config.cloud:
        if not config.deepgram_api_key:
            raise ConfigError("CLOUD mode needs DEEPGRAM_API_KEY for speech-to-text.")
        from livekit.plugins import deepgram

        return deepgram.STT(
            api_key=config.deepgram_api_key,
            model=config.deepgram_model,
            language="en",
            keyterm=["Jarvis", "Hey Jarvis"],
        )

    from jarvis.plugins.whisper_stt import WhisperSTT

    return WhisperSTT(model=config.whisper_model, backend=config.whisper_backend)


# ── LLM ────────────────────────────────────────────────────────────────
def build_llm() -> _llm.LLM:
    if config.cloud:
        # Preference order: Cerebras, then OpenAI.
        if config.cerebras_api_key:
            from livekit.plugins import cerebras

            logger.info("LLM: Cerebras %s", config.cerebras_model)
            return cerebras.LLM(model=config.cerebras_model)
        if config.openai_api_key:
            from livekit.plugins import openai

            logger.info("LLM: OpenAI %s", config.openai_model)
            return openai.LLM(model=config.openai_model)
        raise ConfigError(
            "CLOUD mode needs an LLM key: set CEREBRAS_API_KEY (preferred) or "
            "OPENAI_API_KEY."
        )

    from livekit.plugins import openai

    logger.info("LLM: Ollama %s (local)", config.ollama_model)
    return openai.LLM.with_ollama(
        model=config.ollama_model, base_url=config.ollama_base_url
    )


# ── TTS ────────────────────────────────────────────────────────────────
def build_tts() -> _tts.TTS:
    if config.cloud:
        if not config.deepgram_api_key:
            raise ConfigError("CLOUD mode needs DEEPGRAM_API_KEY for text-to-speech.")
        from livekit.plugins import deepgram

        return deepgram.TTS(
            api_key=config.deepgram_api_key,
            model=config.deepgram_tts_model,
        )

    from jarvis.plugins.kokoro_tts import KokoroTTS

    return KokoroTTS(
        model_path=config.kokoro_model_path,
        voices_path=config.kokoro_voices_path,
        voice=config.kokoro_voice,
    )


# ── Turn detection (optional; local model, both modes) ──────────────────
def build_turn_detection():
    if not config.use_turn_detector:
        return None
    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        return MultilingualModel()
    except Exception as e:  # model not downloaded / plugin missing → VAD-only
        logger.warning("Turn detector unavailable (%s); falling back to VAD.", e)
        return None


def describe() -> str:
    if config.cloud:
        which_llm = "Cerebras" if config.cerebras_api_key else (
            "OpenAI" if config.openai_api_key else "NONE"
        )
        return f"CLOUD (Deepgram STT+TTS + {which_llm})"
    return f"LOCAL (Whisper[{config.whisper_backend}] + Ollama + Kokoro)"
