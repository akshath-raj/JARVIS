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
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=config.endpoint_delay)


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
    elif config.orchestrator == "openai":
        await _entrypoint_openai(ctx)
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
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=config.endpoint_delay),
        # only end the turn after a gap > endpoint_delay; wait longer if the sentence
        # sounds unfinished (up to endpoint_max_delay) so the user can pause to think.
        min_endpointing_delay=config.endpoint_delay,
        max_endpointing_delay=config.endpoint_max_delay,
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


async def _entrypoint_openai(ctx: JobContext) -> None:
    """Cloud brain: the OpenAI Agents SDK (triage + handoffs to Music / Browser /
    Screen specialists) as the LiveKit LLM node. STT/TTS come from the pipeline as
    usual, so the voice stack is unchanged — only the brain is swapped."""
    from jarvis.browser_agent.announcer import Announcer
    from jarvis.openai_agent import OpenAIAgentsLLM, build_brain, describe

    import os

    logger.info("JARVIS starting — voice: %s | brain: %s", pipeline.describe(), describe())
    announcer = Announcer()  # background browser/document tasks speak their result

    # ── Calendar / to-dos / reminders (shared store, read by the HUD) ──────
    tasks = None
    if config.scheduler_enabled:
        from jarvis.scheduler import TaskStore

        tasks = TaskStore(os.path.join(config.memory_dir, "tasks.json"))

    # ── HUD dashboard (hidden; revealed on voice command) ──────────────────
    ui = convlog = open_cb = shared_memory = None
    if config.ui_enabled:
        from jarvis.graph.llm import chat_model
        from jarvis.graph.memory import MemoryStore
        from jarvis.ui import ConversationLog, UIController, UIServer, open_dashboard, wait_until_up

        convlog = ConversationLog(os.path.join(config.memory_dir, "conversations.jsonl"))
        # memory is created here (not inside build_brain) so the UI reads the same one.
        shared_memory = MemoryStore(
            config.memory_dir,
            model=chat_model(config.extract_model, temperature=0.0),
            summarize_every=config.summarize_every,
        )
        ui = UIController(user=config.ui_user, memory=shared_memory, conversations=convlog, tasks=tasks)
        server = UIServer(ui, port=config.ui_port)
        server.start()

        def _open_dashboard() -> None:
            wait_until_up(server.url)
            open_dashboard(server.url, chrome_path=config.browser_chrome_path)

        open_cb = _open_dashboard

    # ── Reminder/alarm scheduler (rings + pops up + speaks at due time) ────
    scheduler = None
    if config.scheduler_enabled:
        from jarvis.scheduler import AlarmScheduler

        scheduler = AlarmScheduler(
            tasks, announce=announcer.announce, ui=ui,
            sound=config.alarm_sound or None,
        )
        if ui is not None:
            def _on_cmd(cmd: dict) -> None:
                c = (cmd or {}).get("cmd")
                if c == "dismiss_alarm":
                    scheduler.dismiss(cmd.get("id"))
                elif c == "complete_todo" and cmd.get("text"):
                    tasks.complete_todo(cmd["text"])
                    scheduler.dismiss(cmd.get("id"))

            ui.set_command_handler(_on_cmd)

    # ── Document RAG (local index) + sandboxed file organiser ──────────────
    rag = organizer = None
    if config.rag_enabled:
        from jarvis.rag import build_rag

        rag = build_rag(config)
    if config.files_enabled:
        from jarvis.files import FileOrganizer, Sandbox

        organizer = FileOrganizer(Sandbox(config.files_sandbox))

    brain, memory = build_brain(
        announce=announcer.announce, memory=shared_memory,
        ui=ui, open_cb=open_cb, scheduler=scheduler, rag=rag, organizer=organizer,
    )

    from jarvis.agents.graph_agent import GraphAgent

    userdata = JarvisContext(activation=_wake(), memory=memory)
    session = AgentSession[JarvisContext](
        userdata=userdata,
        stt=pipeline.build_stt(),
        llm=OpenAIAgentsLLM(brain=brain, memory=memory),
        tts=pipeline.build_tts(),
        turn_detection=pipeline.build_turn_detection(),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=config.endpoint_delay),
        # only end the turn after a gap > endpoint_delay; wait longer if the sentence
        # sounds unfinished (up to endpoint_max_delay) so the user can pause to think.
        min_endpointing_delay=config.endpoint_delay,
        max_endpointing_delay=config.endpoint_max_delay,
    )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = getattr(ev, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", "") or ""
        if role == "user" and text:
            memory.log_turn(text)  # distilled into the durable profile in the background
        if role == "assistant":
            userdata.activation.note_reply(text)
        if convlog is not None and role in ("user", "assistant") and text:
            convlog.add(role, text)  # persistent transcript for the HUD

    announcer.session = session
    if scheduler is not None:
        scheduler.start()  # begin ticking reminders (fires alarms at due time)

    # Warm the model connection in the background so the FIRST real command isn't
    # slow (TLS + first-token cold start otherwise adds several seconds).
    import asyncio as _asyncio

    async def _warmup() -> None:
        try:
            from agents import Runner

            from jarvis.openai_agent.brain import BrainContext

            await Runner.run(brain, input="hello", context=BrainContext(memory=memory), max_turns=1)
        except Exception:
            pass

    _asyncio.create_task(_warmup())

    # Auto-ingest: periodically rescan the watched folders so freshly-downloaded
    # documents get indexed without re-embedding everything.
    if rag is not None and config.rag_scan_seconds > 0:
        import asyncio

        async def _rag_scan_loop() -> None:
            await asyncio.sleep(config.rag_scan_start_delay)  # don't compete with startup
            while True:
                try:
                    # cap new files per cycle so a large backlog never hogs the machine
                    await asyncio.to_thread(rag.sync, None, limit=config.rag_max_per_scan)
                except Exception as e:  # never let indexing crash the assistant
                    logger.warning("RAG scan failed: %s", e)
                await asyncio.sleep(config.rag_scan_seconds)

        asyncio.create_task(_rag_scan_loop())

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
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=config.endpoint_delay),
        # only end the turn after a gap > endpoint_delay; wait longer if the sentence
        # sounds unfinished (up to endpoint_max_delay) so the user can pause to think.
        min_endpointing_delay=config.endpoint_delay,
        max_endpointing_delay=config.endpoint_max_delay,
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
