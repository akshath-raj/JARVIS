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
import re
from dataclasses import dataclass

from agents import Agent, ModelSettings

from jarvis.config import config
from jarvis.graph.memory import MemoryStore
from jarvis.openai_agent.tools import (
    build_browser_tools,
    build_files_tools,
    build_focus_tools,
    build_music_tools,
    build_planner_tools,
    build_reading_tools,
    build_screen_tools,
    build_system_tools,
    build_triage_tools,
    build_ui_tools,
)
from jarvis.tools.browser import BrowserController
from jarvis.tools.media import MediaController
from jarvis.tools.reading_mode import ReadingMode
from jarvis.tools.screen import ScreenController
from jarvis.tools.spotify import SpotifyController
from jarvis.tools.system_settings import SystemSettings
from jarvis.tools.web import TavilyClient

logger = logging.getLogger("jarvis.openai_agent")


@dataclass
class BrainContext:
    """Run context passed to Runner.run — lets tools/instructions reach the memory
    store, and carries the memory block the adapter computes for the current turn."""
    memory: MemoryStore
    memory_block: str = ""
    # The user's original words for this turn. Tools read it to recover intent the
    # model may have dropped when it filled in an argument — e.g. "open ALL my X
    # docs" arrives at open_document as just description="X", losing the "all".
    user_text: str = ""


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
    "playlist/top/liked/recently-played tools. play_song for a song, artist, GENRE or "
    "MOOD ('play some jazz', 'play good jazz music', 'play something upbeat') — it "
    "searches Spotify and plays; play_playlist ONLY when the user names one of THEIR "
    "OWN playlists or says the word 'playlist' ('play my Knight playlist'). Never treat "
    "a genre/mood like 'jazz' as a playlist name. "
    "To CLOSE browser tabs the user has open ('close my VTOP tabs', 'close all youtube "
    "tabs', 'close this tab') use close_tabs (pass what to match, or empty for the "
    "current tab) — it acts on the user's REAL visible Chrome; NEVER use browser_task "
    "to close tabs (it drives a separate profile with none of their tabs). "
    "open_site just opens a website or app's "
    "landing page by name and nothing else. To actually DO something on a website — "
    "search a site and open a result, navigate through pages/menus, log in, fill and "
    "submit a form, book/order, download a file, or read specific account info — use "
    "browser_task with the user's FULL request; it drives a real browser autonomously "
    "(reads the page and clicks/types), so do NOT use open_site for that. "
    "play_youtube whenever the user wants to OPEN, PLAY, WATCH, SHOW or 'pull up' ONE "
    "specific video / trailer / song / clip on YouTube ('open the new Avengers "
    "trailer', 'play the lofi video', 'put on the Interstellar trailer'): call it "
    "ONCE and NOTHING ELSE — do NOT ALSO call open_site or web_search for the same "
    "request, that opens a second unwanted tab (play_youtube already opens the video "
    "in one tab). If instead they ask to LIST / FIND / RECOMMEND / 'good videos on X' "
    "or 'best … videos', use web_search and just TELL them the list — do NOT play "
    "anything. "
    "play_netflix to play a specific show/movie on Netflix ('play <title> on Netflix', "
    "'put on <show>') — it opens and plays it in the user's own logged-in Chrome, so "
    "use it (NOT browser_task) for watching on Netflix; open_site('netflix') only for "
    "the home page. control_video "
    "controls a video already playing (subtitles/speed/volume/brightness/pause/seek). "
    "The LAPTOP'S OWN screen & sound: set_brightness/change_brightness control the "
    "MAC DISPLAY brightness ('dim the screen', 'brightness to 40%', 'brighter'); "
    "set_system_volume/change_system_volume/mute_system control the WHOLE COMPUTER'S "
    "output volume ('turn the volume down', 'mute'), and set_dark_mode toggles Dark "
    "Mode. Keep these DISTINCT from the music tools: when the user is talking about a "
    "SONG that's playing use set_music_volume/change_volume (Spotify); when they mean "
    "the computer, the screen, or nothing is playing, use the system tools. "
    "start_reading_mode when the user wants to READ / 'reading mode' / 'set me up to "
    "read' — it warms the tone, darkens & dims the screen for low glare, and starts "
    "soft background music; stop_reading_mode / 'I'm done reading' restores everything. "
    "Use explain_screen / explain_this ONLY when the user CLEARLY refers to their own "
    "screen or a document they are viewing ('what's on my screen', 'explain this slide/"
    "error/formula', 'read this'); if the request is vague or you're unsure they mean "
    "the screen, ASK what to look at rather than taking a screenshot. To capture AND "
    "read/describe/explain the screen — including 'take a screenshot and tell me what's "
    "there' — ALWAYS use explain_screen; NEVER use take_screenshot for that and NEVER "
    "describe the screen from your own guess. take_screenshot only saves a file and does "
    "not look at anything — use it only when the user just wants a screenshot saved. "
    "explain_this finds "
    "the live document, the page, and reads the surrounding pages of that subtopic. "
    "BUT when the user asks to EXPLAIN or say what a document is about that they NAMED "
    "or JUST OPENED ('explain that document', 'what is this document about', 'explain "
    "the Kalman notes') use summarize_document on that file — it reads the indexed "
    "content reliably; reserve explain_this/explain_screen for what is literally on the "
    "SCREEN (a slide, an error, a chart) when there is no known document to read. "
    "summarize_this summarises the document the user is looking at on screen when they "
    "DON'T name it ('summarise the document in front of me', 'give me the gist of this') "
    "— set section_only=true for just the current section/slide/page. explain_screen, "
    "explain_this and summarize_this always show the screenshot and result on the HUD "
    "automatically. web_search for current/live info (news, "
    "prices, weather, scores) or "
    "for lists of recommendations. remember/forget/recall_about_me for stored personal "
    "facts (NOT listening history). add_reminder/add_todo/add_calendar_event/"
    "complete_todo/stop_alarm for time & tasks (pass absolute ISO datetimes computed "
    "from now). ask_documents for ANY question answered by the user's files (pass a "
    "free-text `document` description + optional `page` — never demand the exact "
    "filename); summarize_document to summarise a file/page; find_document for "
    "where/when a file is. "
    "The user's OWN documents/notes/slides/PDFs/assignments/resume live on their "
    "computer and are opened LOCALLY, never through a browser: for ONE file ('open "
    "my <X>', 'pull up the <X> notes') use open_document; to open EVERY file on a "
    "topic ('open ALL my docs on <X>', 'open everything about <X>', 'open all my <X> "
    "notes') use open_documents. ALWAYS use these for the user's files — NEVER "
    "open_site or browser_task (those are the WEB and would just do a search). Other "
    "file tools organise/move/copy or open by exact path (you can NEVER delete). "
    "show_dashboard / "
    "hide_dashboard / open_dashboard_section / display_on_dashboard for the HUD. "
    "start_focus_mode when the user wants to focus / avoid distractions / do a "
    "pomodoro or deep-work session (it closes distractions and blocks reopening "
    "them); stop_focus_mode ONLY when they clearly say to end focus; focus_status "
    "for how the session is going. "
    "If a request needs MULTIPLE steps, call the tools in sequence and only answer "
    "once you have everything (e.g. 'explain this graph AND check 2026 data' → "
    "explain_this THEN web_search, then give one combined answer). When confirming a "
    "device ACTION (play/pause/open/move/reminder) reply in ONE short sentence; but "
    "when ANSWERING or EXPLAINING (explain_this/explain_screen/summarize_this/"
    "ask_documents/summarize_document/web_search) give the FULL answer — do not compress a detailed "
    "explanation into one sentence. Ask a question only if an essential detail is missing. "
    "To make QUESTIONS / a quiz / flashcards / to test the user on what they are "
    "reading or on a document, first call explain_this (for what's on their screen) or "
    "ask_documents/summarize_document (by name) to get the content, then write the "
    "questions yourself from it. "
    "To REVIEW / critique / suggest improvements / give feedback on the user's own "
    "document(s) ('review my essay', 'what can I improve in my resume', 'pull up both "
    "my resumes and suggest improvements'), FIRST open them with open_document/"
    "open_documents so they're on screen, THEN call review_document once PER document "
    "to get concrete suggestions, and combine them into your reply — do not stop after "
    "merely opening them, and do NOT use ask_documents for this (it only reports what "
    "the text says, it won't critique). "
    "NEVER speak your plan, reasoning, or tool mechanics out loud: do not say things "
    "like 'now I will…', 'let me call…', or 'the user will provide input'. Say nothing "
    "until you have the result, then reply with only the answer or a short confirmation."
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


def _build_screen() -> ScreenController:
    """Screen vision on Cerebras Gemma 4 (fast) when selected + keyed, else OpenAI."""
    if config.agent_provider == "cerebras" and config.cerebras_api_key:
        logger.info("screen vision on Cerebras %s", config.cerebras_vision_model)
        return ScreenController(
            vision_model=config.cerebras_vision_model,
            openai_api_key=config.cerebras_api_key,
            base_url=config.cerebras_base_url,
        )
    return ScreenController(
        vision_model=config.vision_model, openai_api_key=config.openai_api_key
    )


def build_brain(*, spotify=None, browser=None, tavily=None, memory=None,
                media=None, screen=None, announce=None, workspace=None,
                ui=None, open_cb=None, scheduler=None, rag=None, organizer=None,
                focus=None, system=None, reading=None, agent_model=None):
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
    screen = screen or _build_screen()
    if workspace is None:
        from jarvis.documents.workspace import Workspace
        workspace = Workspace(downloads_dir=config.browser_downloads)
    system = system or SystemSettings()
    reading = reading or ReadingMode(
        system=system, spotify=spotify, search_mode=config.spotify_search_mode,
        brightness=config.reading_brightness, volume=config.reading_volume,
        music_query=config.reading_music_query, dark_mode=config.reading_dark_mode,
        night_shift=config.reading_night_shift, wallpaper=config.reading_wallpaper,
    )

    model = agent_model or _resolve_model()  # agent_model lets tests/benchmarks override
    settings = ModelSettings(temperature=0.2)
    if config.agent_provider == "cerebras" and config.cerebras_api_key:
        # gpt-oss is a reasoning model; minimal reasoning keeps tool-calling latency
        # low (the biggest lever on "responses take a long time").
        settings = ModelSettings(temperature=0.2, extra_body={"reasoning_effort": "low"})

    # ONE agent with every tool → a command is a SINGLE model call. Handoffs would
    # add a second sequential call (the routing hop) and double the latency, which
    # is the wrong trade for a voice assistant. `stop_on_first_tool` speaks the
    # tool's return string directly, so there's no extra "phrasing" call either —
    # one round-trip per action, and the model can't narrate an action it didn't do.
    tools = build_music_tools(spotify=spotify, memory=memory, search_mode=config.spotify_search_mode)
    tools += build_browser_tools(
        browser=browser, memory=memory, media=media, announce=announce,
        workspace=workspace,
        browser_agent_enabled=(config.pw_agent_enabled or config.browser_agent_enabled),
        focus=focus, tavily=tavily,
    )
    tools += build_screen_tools(screen=screen, memory=memory, ui=ui, rag=rag)
    tools += build_system_tools(system=system, memory=memory)
    tools += build_reading_tools(reading=reading, memory=memory)
    tools += build_triage_tools(tavily=tavily, memory=memory)
    if ui is not None:
        tools += build_ui_tools(ui=ui, open_cb=open_cb)
    if scheduler is not None:
        tools += build_planner_tools(scheduler=scheduler, memory=memory)
    if rag is not None or organizer is not None:
        tools += build_files_tools(rag=rag, organizer=organizer, memory=memory)
    if focus is not None:
        tools += build_focus_tools(focus=focus, memory=memory)

    agent = Agent[BrainContext](
        name="JARVIS",
        instructions=_instructions,
        model=model,
        model_settings=settings,
        tool_use_behavior=_tool_final_policy,
        tools=tools,
    )
    logger.info("OpenAI Agents brain ready (single agent, model %s, %d tools)", model, len(tools))
    return agent, memory


# Informational/answer tools return DATA the model may need to act on further —
# chain another tool ("explain this AND check 2026 data" → explain_this then
# web_search) or fold several results into one spoken reply. Keep the agent loop
# OPEN after these. Every other tool is a device ACTION whose result is spoken
# directly (one fast round-trip, and the model can't narrate an action it skipped).
_CONTINUE_TOOLS = frozenset({
    "explain_screen", "explain_this", "summarize_this", "ask_documents",
    "summarize_document", "review_document", "find_document", "web_search",
    "list_folder", "recall_about_me",
})

# Opening a document is normally terminal (one round-trip). But when the SAME
# request also asks to review/critique/explain/compare it, we must keep the loop
# open so the model can go on to READ the file and synthesise — "pull up both my
# resumes AND suggest improvements" is one turn, not two.
_OPEN_TOOLS = frozenset({"open_document", "open_documents"})
_ANALYSIS_INTENT = re.compile(
    r"\b(suggest|improv|review|feedback|critiqu|advice|advis|recommend|summar|"
    r"explain|compar|analy|assess|evaluat|rate|score|what|why|how)\b",
    re.I,
)


def _tool_final_policy(run_context, results):
    """Decide, after a turn's tools run, whether to stop or keep reasoning."""
    from agents import ToolsToFinalOutputResult

    if any(r.tool.name in _CONTINUE_TOOLS for r in results):
        return ToolsToFinalOutputResult(is_final_output=False)  # let it chain/synthesise
    # A compound "open X AND review/explain/compare it" — keep going so the model can
    # read the just-opened document(s) and produce the analysis it was asked for.
    if any(r.tool.name in _OPEN_TOOLS for r in results):
        ut = getattr(getattr(run_context, "context", None), "user_text", "") or ""
        if _ANALYSIS_INTENT.search(ut):
            return ToolsToFinalOutputResult(is_final_output=False)
    out = results[-1].output if results else ""
    return ToolsToFinalOutputResult(is_final_output=True, final_output=str(out))


def describe() -> str:
    on_cerebras = config.agent_provider == "cerebras" and config.cerebras_api_key
    brain = f"Cerebras {config.cerebras_model}" if on_cerebras else f"OpenAI {config.cloud_agent_model}"
    vision = config.cerebras_vision_model if on_cerebras else config.vision_model
    return f"OpenAI Agents SDK (single agent, {brain}, vision {vision}, RAG on {brain})"
