"""Graph wiring tests with a stubbed chat model (no Ollama).

Validates the custom graph's control flow deterministically: a tool call routes
agent→tools→speak (and TTS gets the tool's return), a plain answer streams from
the agent node, and an empty answer hits the fallback node.
"""
from __future__ import annotations

import asyncio
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

from jarvis.graph.build import build_graph
from jarvis.graph.memory import MemoryStore
from jarvis.tools.spotify import PlayResult


class FakeChat:
    """Stands in for ChatOllama. `router(text) -> AIMessage` decides each turn."""

    def __init__(self, router):
        self._router = router

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                text = str(m.content)
                break
        return self._router(text)


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def _run(router, tmp_path, user_text):
    rec = []
    sp = mock.MagicMock()
    sp.play_query.side_effect = lambda q, m, l=False: rec.append(("play_song", q)) or PlayResult(
        label=f"{q} by Artist", source="web"
    )
    br = mock.MagicMock()
    br.open_site.side_effect = lambda n: rec.append(("open_site", n)) or f"opened {n}"
    tv = mock.MagicMock()
    tv.search.side_effect = lambda q, *a, **k: rec.append(("web_search", q)) or "fresh news"
    memory = MemoryStore(str(tmp_path))

    graph, mem = build_graph(spotify=sp, browser=br, tavily=tv, memory=memory, model=FakeChat(router))
    out = asyncio.run(graph.ainvoke({"messages": [HumanMessage(user_text)]}))
    spoken = out["messages"][-1].content
    return (rec[0][0] if rec else "DIRECT"), spoken, mem


def test_tool_call_routes_and_speaks_result(tmp_path):
    router = lambda t: _tool_call("play_song", {"query": "despacito"})
    route, spoken, mem = _run(router, tmp_path, "play despacito")
    assert route == "play_song"
    assert spoken == "now playing despacito by Artist"      # tool result spoken directly
    # Activity goes to the transient learning log (distilled into the profile later),
    # not the durable profile itself.
    assert "played music: despacito" in [e["detail"] for e in mem._read_log()]


def test_browser_tool(tmp_path):
    router = lambda t: _tool_call("open_site", {"name": "instagram"})
    route, spoken, _ = _run(router, tmp_path, "open instagram")
    assert route == "open_site" and spoken == "opened instagram"


def test_web_search_tool(tmp_path):
    router = lambda t: _tool_call("web_search", {"query": "AI news"})
    route, spoken, _ = _run(router, tmp_path, "latest news")
    assert route == "web_search" and spoken == "fresh news"


def test_plain_answer_no_tool(tmp_path):
    router = lambda t: AIMessage(content="Leonardo da Vinci painted the Mona Lisa.")
    route, spoken, _ = _run(router, tmp_path, "who painted the mona lisa")
    assert route == "DIRECT"
    assert spoken == "Leonardo da Vinci painted the Mona Lisa."


def test_empty_answer_hits_fallback(tmp_path):
    router = lambda t: AIMessage(content="")   # model said nothing, no tool
    route, spoken, _ = _run(router, tmp_path, "mumble")
    assert route == "DIRECT"
    assert "say it again" in spoken.lower()     # fallback keeps JARVIS from going silent


def test_remember_tool_writes_memory(tmp_path):
    router = lambda t: _tool_call("remember", {"text": "loves sushi"})
    route, spoken, mem = _run(router, tmp_path, "remember I love sushi")
    assert "loves sushi" in mem.all()
    assert spoken.startswith("noted")
