"""JARVIS entrypoint — local (or cloud) voice assistant.

Voice modes (JARVIS_MODE): 0 = LOCAL (Whisper + Ollama + Kokoro), 1 = CLOUD.

Brain (JARVIS_ORCHESTRATOR):
  * "langgraph" (default) — a LangGraph react-style graph with all tools + a
    persistent, personalising memory store, wrapped by VoiceLLMAdapter.
  * "native" — the previous hand-rolled LiveKit agent (fallback / rollback).

Run locally in your terminal (uses your mic + speakers, no LiveKit cloud):

    python -m jarvis.agent console
"""
from __future__ import annotations

import logging

from livekit.agents import AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero

from jarvis import pipeline
from jarvis.activation import WakeController
from jarvis.config import config
from jarvis.context import JarvisContext

logger = logging.getLogger("jarvis")


def prewarm(proc: JobProcess) -> None:
    # Load VAD once per worker process.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)


def _wake() -> WakeController:
    return WakeController(
        enabled=config.wake_enabled,
        words=config.wake_words,
        followup_seconds=config.wake_followup_seconds,
        continuation_seconds=config.wake_continuation_seconds,
    )


async def entrypoint(ctx: JobContext) -> None:
    if config.orchestrator == "native":
        await _entrypoint_native(ctx)
    else:
        await _entrypoint_langgraph(ctx)


async def _entrypoint_langgraph(ctx: JobContext) -> None:
    from jarvis.agents.graph_agent import GraphAgent
    from jarvis.browser_agent.announcer import Announcer
    from jarvis.graph import VoiceLLMAdapter, build_graph, describe

    logger.info("JARVIS starting — voice: %s | brain: %s", pipeline.describe(), describe())
    announcer = Announcer()  # the browser agent speaks its result when a bg task finishes
    graph, memory = build_graph(announce=announcer.announce)

    userdata = JarvisContext(activation=_wake(), memory=memory)
    session = AgentSession[JarvisContext](
        userdata=userdata,
        stt=pipeline.build_stt(),
        llm=VoiceLLMAdapter(graph=graph),
        tts=pipeline.build_tts(),
        turn_detection=pipeline.build_turn_detection(),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=0.4),
    )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = getattr(ev, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        # Not every conversation item is a message (e.g. AgentHandoff has no
        # text_content) — access it defensively.
        text = getattr(item, "text_content", "") or ""
        if role == "user" and text:
            # Feed the transient learning log; the memory store distills it into
            # the durable profile in the background every N entries (see memory.py).
            memory.log_turn(text)
        if role == "assistant":
            userdata.activation.note_reply(text)

    announcer.session = session  # now background browser tasks can speak
    await session.start(agent=GraphAgent(), room=ctx.room)


async def _entrypoint_native(ctx: JobContext) -> None:
    """Previous architecture: a single hand-rolled agent with all tools."""
    from openai import OpenAI

    from jarvis.agents import JarvisAgent
    from jarvis.specialists import BrowserSpecialist
    from jarvis.tools.browser import BrowserController
    from jarvis.tools.spotify import SpotifyController
    from jarvis.tools.web import TavilyClient

    logger.info("JARVIS starting (native) — voice: %s", pipeline.describe())
    tool_client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
    tavily = TavilyClient(config.tavily_api_key)
    browser = BrowserSpecialist(
        client=tool_client,
        model=config.tool_model,
        browser=BrowserController(config.browser_app),
        tavily=tavily,
    )
    userdata = JarvisContext(
        spotify=SpotifyController(config.spotify_client_id, config.spotify_client_secret),
        activation=_wake(),
        browser=browser,
        tavily=tavily,
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

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = getattr(ev, "item", None)
        if item is not None and getattr(item, "role", None) == "assistant":
            userdata.activation.note_reply(getattr(item, "text_content", "") or "")

    await session.start(agent=JarvisAgent(), room=ctx.room)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
