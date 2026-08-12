"""Assemble the JARVIS OpenAI Agents SDK brain: a triage agent that HANDS OFF to
specialist agents, all on a frontier OpenAI model.

    ┌─────────────┐   handoff   ┌──────────────┐
    │   JARVIS    ├────────────▶│  Music agent │  (Spotify)
    │  (triage)   ├────────────▶│ Browser agent│  (Chrome / video / tasks / docs)
    │  facts +    ├────────────▶│ Screen agent │  (screenshot + vision)
    │ web + memory│             └──────────────┘
    └─────────────┘

Handoffs are reliable here because a frontier model is driving (unlike local
models, where handoffs misfire — hence the LangGraph brain stays single-agent).
The specialists reuse the exact same controllers as the local brain, so every
music / browser / document capability carries over, plus screen understanding.

`BrainContext` carries the memory store and the per-turn memory block; the triage
agent's dynamic instructions inject the most relevant memories each turn so the
cloud brain personalises just like the local one.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from agents import Agent, ModelSettings

from jarvis.config import config
from jarvis.graph.memory import MemoryStore
from jarvis.openai_agent.tools import (
    build_browser_tools,
    build_files_tools,
    build_music_tools,
    build_planner_tools,
    build_screen_tools,
    build_triage_tools,
    build_ui_tools,
)
from jarvis.tools.browser import BrowserController
from jarvis.tools.media import MediaController
from jarvis.tools.screen import ScreenController
from jarvis.tools.spotify import SpotifyController
from jarvis.tools.web import TavilyClient

logger = logging.getLogger("jarvis.openai_agent")


@dataclass
class BrainContext:
    """Run context passed to Runner.run — lets tools/instructions reach the memory
    store, and carries the memory block the adapter computes for the current turn."""
    memory: MemoryStore
    memory_block: str = ""


_PERSONA = (
    "You are JARVIS, a concise, witty British butler. Replies are spoken aloud, so "
    "answer in ONE short sentence with no markdown, no follow-up questions, and no "
    "filler. Answer well-known or historical facts yourself from your own knowledge "
    "with NO tool and NO handoff. Address the user as 'sir' occasionally."
)

# One agent, one API call per turn (handoffs would double latency for voice). The
# tool docstrings carry the detail; this just steers the tricky choices. The
# cardinal rule: DO the action by CALLING the tool — never narrate a done action.
_GUIDE = (
    " CARDINAL RULE: to DO anything you MUST call its tool. You have NOT paused, "
    "resumed, skipped, changed the volume, added a reminder, moved a file, etc. "
    "unless you actually CALLED that tool — never say you did something you didn't "
    "call. Answer general-knowledge/historical facts yourself with no tool. "
    "Music (Spotify): pause_music, resume_music, next_song, play_song, "
    "set_music_volume (exact number) or change_volume (up/down), set_loop, the "
    "playlist/top/liked/recently-played tools. open_site opens a website or app by "
    "name. play_youtube ONLY when the user says PLAY or WATCH one specific video; if "
    "they ask to LIST / FIND / RECOMMEND / 'good videos on X' or 'best … videos', use "
    "web_search and just TELL them the list — do NOT play anything. control_video "
    "controls a video already playing (subtitles/speed/volume/brightness/pause/seek). "
    "explain_screen to look at what's on screen (show_in_ui=true to also render it on "
    "the HUD). web_search for current/live info (news, prices, weather, scores) or "
    "for lists of recommendations. remember/forget/recall_about_me for stored personal "
    "facts (NOT listening history). add_reminder/add_todo/add_calendar_event/"
    "complete_todo/stop_alarm for time & tasks (pass absolute ISO datetimes computed "
    "from now). ask_documents for questions answered by the user's files; the file "
    "tools to organise/move/copy/open (you can NEVER delete). show_dashboard / "
    "hide_dashboard / open_dashboard_section / display_on_dashboard for the HUD. Reply "
    "in ONE short sentence; ask a question only if an essential detail is missing."
)


def _date_note() -> str:
    now = datetime.datetime.now()
    return (
        f" The current date and time is {now:%A, %-d %B %Y, %-I:%M %p}. Use this to "
        "resolve relative times like 'in an hour', 'tonight', or '9pm today' — when "
        "setting a reminder/event, compute the absolute time from now and pass it as "
        "an ISO 8601 datetime. Your training knowledge is out of date, so for "
        "anything time-sensitive trust tool results, not your memory, and assume the "
        "present year; do not add a year to a web search unless the user said one."
    )


def _instructions(run_context, agent) -> str:
    ctx = getattr(run_context, "context", None)
    block = getattr(ctx, "memory_block", "") if ctx else ""
    return _PERSONA + _GUIDE + _date_note() + (block or "")


def _resolve_model():
    """Return the model the agent runs on. Cerebras (OpenAI-compatible, far faster)
    when selected AND its key is present; otherwise the OpenAI model string."""
    if config.agent_provider == "cerebras" and config.cerebras_api_key:
        from agents import OpenAIChatCompletionsModel, set_tracing_disabled
        from openai import AsyncOpenAI

        set_tracing_disabled(True)  # don't ship traces to OpenAI for a non-OpenAI provider
        client = AsyncOpenAI(base_url=config.cerebras_base_url, api_key=config.cerebras_api_key)
        logger.info("agent brain on Cerebras %s", config.cerebras_model)
        return OpenAIChatCompletionsModel(model=config.cerebras_model, openai_client=client)
    return config.cloud_agent_model


def build_brain(*, spotify=None, browser=None, tavily=None, memory=None,
                media=None, screen=None, announce=None, workspace=None,
                ui=None, open_cb=None, scheduler=None, rag=None, organizer=None,
                agent_model=None):
    """Return (triage_agent, memory). Dependencies are injectable for tests.

    `ui` (a UIController) enables the HUD dashboard tools; `open_cb` is called to
    launch the browser window when the user asks to show the dashboard."""
    if memory is None:
        from jarvis.graph.llm import chat_model
        memory = MemoryStore(
            config.memory_dir,
            model=chat_model(config.extract_model, temperature=0.0),
            summarize_every=config.summarize_every,
        )
    spotify = spotify or SpotifyController(config.spotify_client_id, config.spotify_client_secret)
    browser = browser or BrowserController(config.browser_app)
    tavily = tavily or TavilyClient(config.tavily_api_key)
    media = media or MediaController(config.browser_app)
    screen = screen or ScreenController(
        vision_model=config.vision_model, openai_api_key=config.openai_api_key
    )
    if workspace is None:
        from jarvis.documents.workspace import Workspace
        workspace = Workspace(downloads_dir=config.browser_downloads)

    model = agent_model or _resolve_model()  # agent_model lets tests/benchmarks override
    settings = ModelSettings(temperature=0.2)

    # ONE agent with every tool → a command is a SINGLE model call. Handoffs would
    # add a second sequential call (the routing hop) and double the latency, which
    # is the wrong trade for a voice assistant. `stop_on_first_tool` speaks the
    # tool's return string directly, so there's no extra "phrasing" call either —
    # one round-trip per action, and the model can't narrate an action it didn't do.
    tools = build_music_tools(spotify=spotify, memory=memory, search_mode=config.spotify_search_mode)
    tools += build_browser_tools(
        browser=browser, memory=memory, media=media, announce=announce,
        workspace=workspace, browser_agent_enabled=config.browser_agent_enabled,
    )
    tools += build_screen_tools(screen=screen, memory=memory, ui=ui)
    tools += build_triage_tools(tavily=tavily, memory=memory)
    if ui is not None:
        tools += build_ui_tools(ui=ui, open_cb=open_cb)
    if scheduler is not None:
        tools += build_planner_tools(scheduler=scheduler, memory=memory)
    if rag is not None or organizer is not None:
        tools += build_files_tools(rag=rag, organizer=organizer, memory=memory)

    agent = Agent[BrainContext](
        name="JARVIS",
        instructions=_instructions,
        model=model,
        model_settings=settings,
        tool_use_behavior="stop_on_first_tool",
        tools=tools,
    )
    logger.info("OpenAI Agents brain ready (single agent, model %s, %d tools)", model, len(tools))
    return agent, memory


def describe() -> str:
    on_cerebras = config.agent_provider == "cerebras" and config.cerebras_api_key
    brain = f"Cerebras {config.cerebras_model}" if on_cerebras else f"OpenAI {config.cloud_agent_model}"
    return f"OpenAI Agents SDK (single agent, {brain}, vision {config.vision_model})"
