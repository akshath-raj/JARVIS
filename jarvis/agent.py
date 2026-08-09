"""JARVIS entrypoint — local (or cloud) voice assistant.

Modes (JARVIS_MODE): 0 = LOCAL (Whisper + Ollama + Kokoro), 1 = CLOUD
(Deepgram STT+TTS + Cerebras/OpenAI).

A single JarvisAgent answers questions and controls Spotify. Wake word ("jarvis"
/ "hey jarvis") gates the session with smart follow-up (toggle JARVIS_WAKE=0).
Shared state (Spotify controller, wake state) lives in session.userdata.

Run locally in your terminal (uses your mic + speakers, no LiveKit cloud):

    python -m jarvis.agent console
"""
from __future__ import annotations

import logging

from livekit.agents import AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero

from jarvis import pipeline
from jarvis.activation import WakeController
from jarvis.agents import JarvisAgent
from jarvis.config import config
from jarvis.context import JarvisContext
from jarvis.tools.spotify import SpotifyController

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
        activation=WakeController(
            enabled=config.wake_enabled,
            words=config.wake_words,
            followup_seconds=config.wake_followup_seconds,
        ),
        search_mode=config.spotify_search_mode,
    )

    session = AgentSession[JarvisContext](
        userdata=userdata,
        stt=pipeline.build_stt(),
        llm=pipeline.build_llm(),
        tts=pipeline.build_tts(),
        turn_detection=pipeline.build_turn_detection(),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=0.4),
    )

    # After each JARVIS reply, stay awake iff it was a clarification question.
    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = getattr(ev, "item", None)
        if item is not None and getattr(item, "role", None) == "assistant":
            userdata.activation.note_reply(item.text_content or "")

    # JarvisAgent.on_enter delivers the greeting.
    await session.start(agent=JarvisAgent(), room=ctx.room)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
