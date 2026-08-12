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

_ROUTING = (
    " Hand off to the Music agent for ANYTHING about playing or controlling Spotify "
    "or the user's music library — play, pause, resume/continue the music, skip, "
    "volume, loop/repeat AND stop looping, playlists, and their top / liked / "
    "recently-played songs and artists. Hand off to the Browser agent to open "
    "websites/apps, control a YouTube/Netflix VIDEO the user is watching "
    "(subtitles/captions, speed, volume, brightness, pause/seek/fullscreen, close "
    "the tab), run multi-step logged-in web tasks (downloading files, checking "
    "accounts like VTOP/AWS/Amazon), or work on documents/assignments. Hand off to "
    "the Screen agent ONLY to look at / read / explain what is CURRENTLY DISPLAYED "
    "on the user's screen — never to control a video. Hand off to the Planner agent "
    "for the calendar, to-do list, and time-based reminders/alarms — 'remind me to X "
    "at 9pm / in an hour', 'add X to my to-do list', 'I'm done with X', 'what's on my "
    "calendar', 'stop the alarm', or 'what time is it'. Use web_search yourself only "
    "when the answer depends on current/live info (news, prices, weather, scores). "
    "recall_about_me is ONLY for personal facts you've stored about the user — never "
    "their listening history (that's the Music agent's recently_played). Use "
    "remember/forget for saving/removing such facts. For the on-screen dashboard/HUD: "
    "show_dashboard to bring up / open the interface, hide_dashboard to dismiss it, "
    "open_dashboard_section to pull up the user's profile/memories/past "
    "conversations on it, and display_on_dashboard to show a text answer on it. When "
    "the user asks to SHOW or EXPLAIN what's on their screen ON the dashboard/UI, "
    "hand off to the Screen agent (it renders the analysis on the HUD). Always "
    "perform the action immediately; never just claim you did, and never ask for a "
    "detail the user already gave."
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


def _triage_instructions(run_context, agent) -> str:
    ctx = getattr(run_context, "context", None)
    block = getattr(ctx, "memory_block", "") if ctx else ""
    return _PERSONA + _ROUTING + _date_note() + (block or "")


def build_brain(*, spotify=None, browser=None, tavily=None, memory=None,
                media=None, screen=None, announce=None, workspace=None,
                ui=None, open_cb=None, scheduler=None):
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

    model = config.cloud_agent_model
    settings = ModelSettings(temperature=0.2)

    music_agent = Agent[BrainContext](
        name="Music",
        handoff_description="Plays and controls Spotify music and the user's library.",
        instructions=(
            _PERSONA + " You control Spotify. Call the matching tool immediately for "
            "the user's request (play/pause/resume/skip/volume/loop and stop-looping/"
            "playlists/top/liked/recently-played). Use the user's own words as the "
            "search query, and EXTRACT any playlist name they gave (e.g. 'add this to "
            "my workout playlist' → add_current_song_to_playlist('workout')). For "
            "their listening history use recently_played. Only ask a question if an "
            "essential detail is genuinely missing; then confirm in one short "
            "sentence." + _date_note()
        ),
        model=model,
        model_settings=settings,
        tools=build_music_tools(spotify=spotify, memory=memory, search_mode=config.spotify_search_mode),
    )

    browser_agent = Agent[BrainContext](
        name="Browser",
        handoff_description=(
            "Opens websites/apps, plays and controls YouTube/Netflix video, runs "
            "multi-step logged-in web tasks, and handles documents/assignments."
        ),
        instructions=(
            _PERSONA + " You drive the browser and on-screen video, run logged-in web "
            "tasks, and handle documents. Use open_site to open a site by name; "
            "play_youtube only when the user names something to watch; browser_task "
            "for multi-step logged-in jobs (downloads, account info) — NOT for merely "
            "opening a site. control_video controls the video the user is watching — "
            "subtitles/captions, speed, volume, brightness, pause, seek, fullscreen "
            "(use play, not restart, to resume). For 'download X and explain it' call "
            "download_and_explain ALONE (it downloads AND reads it — do NOT also call "
            "browser_task). If the user says to finish/do the assignment (optionally "
            "naming a format like a notebook/doc/slides), call do_assignment right "
            "away with their words — the assignment is already loaded, don't ask for "
            "its content. To open the finished answer you MUST call open_answer — "
            "never say the answer is open unless you actually called it. In general "
            "you have NOT performed an action unless you called its tool. Confirm in "
            "one short sentence." + _date_note()
        ),
        model=model,
        model_settings=settings,
        tools=build_browser_tools(
            browser=browser, memory=memory, media=media, announce=announce,
            workspace=workspace, browser_agent_enabled=config.browser_agent_enabled,
        ),
    )

    screen_agent = Agent[BrainContext](
        name="Screen",
        handoff_description=(
            "Reads and explains what is currently DISPLAYED on the user's screen. "
            "Not for controlling video (that's the Browser agent)."
        ),
        instructions=(
            "You are JARVIS. The user wants to know about what's on their screen. Call "
            "explain_screen with their question. If they asked to SHOW / OPEN / DISPLAY "
            "the explanation on the dashboard / UI / screen (e.g. 'explain this formula "
            "and open it in the UI'), pass show_in_ui=true so it renders on the HUD. "
            "Then relay the returned explanation to the user IN FULL and in detail — do "
            "not summarise or shorten it. It is spoken aloud, so use plain prose."
        ),
        model=model,
        model_settings=settings,
        tools=build_screen_tools(screen=screen, memory=memory, ui=ui),
    )

    handoffs = [music_agent, browser_agent, screen_agent]
    if scheduler is not None:
        planner_agent = Agent[BrainContext](
            name="Planner",
            handoff_description=(
                "Manages the user's calendar, to-do list, and time-based reminders/"
                "alarms; tells the time; stops a ringing alarm."
            ),
            instructions=(
                _PERSONA + " You manage the user's time: to-dos, calendar events, and "
                "reminders that ring an alarm. add_reminder for 'remind me to X at/in "
                "…' (it rings an alarm at that time); add_todo for a task with no time; "
                "add_calendar_event to schedule something; complete_todo when they say "
                "they've finished something ('I'm done with X'); stop_alarm to silence a "
                "ringing alarm. When setting a reminder or event, compute the absolute "
                "time from the current time and pass it as an ISO 8601 datetime. Confirm "
                "in one short sentence." + _date_note()
            ),
            model=model,
            model_settings=settings,
            tools=build_planner_tools(scheduler=scheduler, memory=memory),
        )
        handoffs.append(planner_agent)

    triage_tools = build_triage_tools(tavily=tavily, memory=memory)
    if ui is not None:
        triage_tools = triage_tools + build_ui_tools(ui=ui, open_cb=open_cb)

    triage = Agent[BrainContext](
        name="JARVIS",
        instructions=_triage_instructions,
        model=model,
        model_settings=settings,
        tools=triage_tools,
        handoffs=handoffs,
    )

    logger.info(
        "OpenAI Agents brain ready (model %s; specialists: Music, Browser, Screen)",
        model,
    )
    return triage, memory


def describe() -> str:
    return (
        f"OpenAI Agents SDK (triage+handoffs, model {config.cloud_agent_model}, "
        f"vision {config.vision_model})"
    )
