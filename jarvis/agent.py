"""JARVIS entrypoint — fully-local, multi-agent voice assistant.

Pipeline:  mic -> Silero VAD -> Whisper (STT) -> qwen3 via Ollama (LLM)
           -> Kokoro (TTS) -> speaker

Agents:    RouterAgent (coordinator) hands off to ChatAgent (general Q&A) or
           MusicAgent (Spotify). Specialists can hand off to each other. The
           shared SpotifyController lives in session.userdata.

Run locally in your terminal (uses your mic + speakers, no LiveKit cloud):

    python -m jarvis.agent console
"""
from __future__ import annotations

import logging

from livekit.agents import AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

from jarvis.agents import RouterAgent
from jarvis.config import config
from jarvis.context import JarvisContext
from jarvis.plugins.kokoro_tts import KokoroTTS
from jarvis.plugins.whisper_stt import WhisperSTT
from jarvis.tools.spotify import SpotifyController

logger = logging.getLogger("jarvis")


async def entrypoint(ctx: JobContext) -> None:
    userdata = JarvisContext(
        spotify=SpotifyController(
            config.spotify_client_id, config.spotify_client_secret
        ),
        search_mode=config.spotify_search_mode,
    )

    session = AgentSession[JarvisContext](
        userdata=userdata,
        stt=WhisperSTT(model=config.whisper_model),
        llm=openai.LLM.with_ollama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        ),
        tts=KokoroTTS(
            model_path=config.kokoro_model_path,
            voices_path=config.kokoro_voices_path,
            voice=config.kokoro_voice,
        ),
        vad=silero.VAD.load(min_silence_duration=0.4),
    )

    # RouterAgent.on_enter delivers the greeting.
    await session.start(agent=RouterAgent(), room=ctx.room)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
