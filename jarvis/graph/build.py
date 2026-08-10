"""Assemble the JARVIS LangGraph brain: a single react-agent with all tools,
personalised each turn by the long-term memory store.

We use one `create_react_agent` with every tool bound (rather than a multi-agent
supervisor) because the feasibility spike showed local models handle a single
scoped react loop reliably and fast, whereas handoffs are slow and make the model
speak plumbing / narrate instead of calling tools.
"""
from __future__ import annotations

import datetime
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from jarvis.config import config
from jarvis.graph.llm import chat_model
from jarvis.graph.memory import MemoryStore
from jarvis.graph.tools import build_tools
from jarvis.tools.browser import BrowserController
from jarvis.tools.media import MediaController
from jarvis.tools.spotify import SpotifyController
from jarvis.tools.web import TavilyClient

logger = logging.getLogger("jarvis.graph")

# Persona + routing. Lean and behavioural (local models narrate if you over-script).
PERSONA = (
    "You are JARVIS, a concise, witty British butler. Reply in ONE very short "
    "sentence, no follow-up questions, no filler like 'Enjoy!' or 'Would you "
    "like…'. Answer well-known or historical facts yourself from your own "
    "knowledge with NO tool — never open a browser tab to answer a factual "
    "question. Use the Spotify tools for music. Use open_site to open a website or "
    "app BY NAME (e.g. 'open youtube', 'open instagram'); use play_youtube ONLY "
    "when the user names something to watch/play; latest_channel_video for a "
    "creator's newest video; open_reels/open_shorts for short-video feeds. Use "
    "web_search only when the answer depends on current/live info (news, prices, "
    "weather, scores). Use browser_task for multi-step jobs on a website that need "
    "logging in, navigating, downloading files, or reading account info (e.g. "
    "download materials from VTOP, check an AWS balance) — NOT for merely opening a "
    "site. To download a document/assignment AND explain it, use "
    "download_and_explain; to complete the loaded assignment with an AI and save it "
    "(as a notebook/doc/slides/script), use do_assignment; to open the finished "
    "answer for review, use open_answer. To control a video the user is WATCHING in "
    "Chrome (YouTube/Netflix) — volume, playback speed, brightness, subtitles, "
    "pause/resume, seek, fullscreen — use control_video; use close_tab to close the "
    "current tab. To resume/continue a paused video use control_video play, NOT "
    "restart. Use remember when told to remember "
    "something, forget to remove a memory, recall_about_me for 'what do you know "
    "about me'. Always "
    "actually CALL the matching tool to perform an action; never just say you did "
    "it. Address the user as 'sir' occasionally. /no_think"
)


def _date_note() -> str:
    t = datetime.date.today()
    return (
        f" Today's actual date is {t:%A}, {t.day} {t:%B} {t.year}. Your own training "
        "knowledge is out of date, so for anything time-sensitive — news, prices, "
        "scores, or the 'latest/upcoming/current' anything — trust the tool results, "
        "not your memory, and assume the present year. When you search the web or use "
        "the browser, do NOT add a year to the query unless the user explicitly said "
        "one; just search for the latest/current."
    )


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return str(getattr(m, "content", ""))
    return ""


def build_graph(*, spotify=None, browser=None, tavily=None, memory=None, model=None, announce=None):
    """Return (compiled_graph, memory). Fully local.

    Dependencies are injectable so tests can pass stubbed controllers / a
    non-embedding memory and a fake model. `announce` lets the browser agent
    speak its result when a background task finishes.
    """
    if memory is None:
        memory = MemoryStore(
            config.memory_dir,
            model=chat_model(config.extract_model, temperature=0.0),
            summarize_every=config.summarize_every,
        )
    from jarvis.documents.workspace import Workspace
    workspace = Workspace(downloads_dir=config.browser_downloads)
    tools = build_tools(
        spotify=spotify or SpotifyController(config.spotify_client_id, config.spotify_client_secret),
        browser=browser or BrowserController(config.browser_app),
        tavily=tavily or TavilyClient(config.tavily_api_key),
        memory=memory,
        search_mode=config.spotify_search_mode,
        announce=announce,
        workspace=workspace,
        media=MediaController(config.browser_app),
    )

    model_with_tools = (model or chat_model(config.ollama_model, temperature=0.1)).bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent(state: MessagesState) -> dict:
        # Inject the memories most relevant to the current request → personalisation.
        block = memory.prompt_block(_last_user_text(state["messages"]), limit=5)
        resp = await model_with_tools.ainvoke([SystemMessage(PERSONA + _date_note() + block), *state["messages"]])
        return {"messages": [resp]}

    def speak(state: MessagesState) -> dict:
        # Speak the tool result(s) directly — no second LLM call. Our tools already
        # return clean, speakable strings, so this halves action latency and avoids
        # the model re-narrating (double-speak). Streamed via the "custom" channel
        # because a hand-built message isn't emitted in "messages" stream mode.
        outs = []
        for m in reversed(state["messages"]):
            if isinstance(m, ToolMessage):
                outs.append(str(m.content))
            else:
                break
        text = " ".join(reversed(outs)).strip() or "Done, sir."
        get_stream_writer()(text)
        return {"messages": [AIMessage(content=text)]}

    def fallback(state: MessagesState) -> dict:
        # The model produced neither a tool call nor any spoken text — never leave
        # the user in silence.
        msg = "Sorry sir, I didn't catch that — could you say it again?"
        get_stream_writer()(msg)
        return {"messages": [AIMessage(content=msg)]}

    def route(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        if not str(getattr(last, "content", "") or "").strip():
            return "fallback"  # empty answer → say something
        return END  # streamed fact answer

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", tool_node)
    g.add_node("speak", speak)
    g.add_node("fallback", fallback)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "fallback": "fallback", END: END})
    g.add_edge("tools", "speak")
    g.add_edge("speak", END)
    g.add_edge("fallback", END)
    graph = g.compile()

    logger.info("LangGraph brain ready (%d tools, memory at %s)", len(tools), config.memory_dir)
    return graph, memory


def describe() -> str:
    return f"LangGraph react-agent + memory (Ollama {config.ollama_model}, embed {config.embed_model})"
