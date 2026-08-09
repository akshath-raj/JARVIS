"""JARVIS entrypoint — fully-local (or cloud) multi-agent voice assistant.

Modes (JARVIS_MODE): 0 = LOCAL (Whisper + Ollama + Kokoro), 1 = CLOUD
(Deepgram + Cerebras/OpenAI + Cartesia).

Agents: RouterAgent (coordinator) hands off to ChatAgent, MusicAgent,
CalendarAgent, or FileAgent; specialists hand back to the coordinator to
re-route. Shared state (Spotify controller, wake gate) lives in session.userdata.

Wake word "Hey Jarvis" gates the whole session (toggle with JARVIS_WAKE=0).

Run locally in your terminal (uses your mic + speakers, no LiveKit cloud):

    python -m jarvis.agent console
"""
from __future__ import annotations

import logging

from livekit.agents import AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero

from jarvis import pipeline
from jarvis.agents import RouterAgent
from jarvis.config import config
from jarvis.context import JarvisContext
from jarvis.relevance import RelevanceGate
from jarvis.tools.spotify import SpotifyController
from jarvis.wake import WakeGate

logger = logging.getLogger("jarvis")


def prewarm(proc: JobProcess) -> None:
    # Load VAD once per worker process.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)


async def entrypoint(ctx: JobContext) -> None:
    logger.info("JARVIS starting — mode: %s", pipeline.describe())

    userdata = JarvisContext(
        spotify=SpotifyController(
            config.spotify_client_id, config.spotify_client_secret
        ),
        wake=WakeGate(
            enabled=config.wake_enabled,
            model=config.wake_model,
            threshold=config.wake_threshold,
            active_seconds=config.wake_active_seconds,
        ),
        relevance=RelevanceGate(
            enabled=config.relevance_enabled,
            model=config.relevance_model,
            base_url=config.ollama_base_url,
        ),
        search_mode=config.spotify_search_mode,
        local_llm=pipeline.local_action_llm(),
    )

    session = AgentSession[JarvisContext](
        userdata=userdata,
        stt=pipeline.build_stt(),
        llm=pipeline.build_llm(),
        tts=pipeline.build_tts(),
        turn_detection=pipeline.build_turn_detection(),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=0.4),
    )

    # RouterAgent.on_enter delivers the greeting.
    await session.start(agent=RouterAgent(), room=ctx.room)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
