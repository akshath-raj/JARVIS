"""Tests for the specialist tool-calling loop (base.Specialist).

The Ollama/OpenAI client is stubbed so no model runs: we feed canned tool_calls
and assert the loop dispatches to the right scoped tool and returns its string.
This proves delegation works deterministically, independent of the local model.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from jarvis.specialists.base import Specialist, ToolSpec, no_params


def _tool_call(cid, name, arguments):
    return NS(id=cid, type="function", function=NS(name=name, arguments=arguments))


def _response(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class FakeClient:
    """Minimal stand-in for openai.OpenAI: returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def _specialist(client, tools, chain=False):
    class S(Specialist):
        pass

    S.chain = chain
    return S(client=client, model="stub", tools=tools)


def test_dispatches_single_tool():
    log = {}
    tools = [
        ToolSpec("open_site", "open a site", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                 lambda name: log.update(name=name) or f"opened {name}"),
    ]
    client = FakeClient([_response(tool_calls=[_tool_call("1", "open_site", '{"name": "instagram"}')])])
    out = _specialist(client, tools).run("open instagram")
    assert out == "opened instagram"
    assert log["name"] == "instagram"
    assert client.calls == 1  # single LLM hop for a single-tool request


def test_compound_request_runs_all_calls_in_one_round():
    tools = [
        ToolSpec("open_site", "open", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                 lambda name: f"opened {name}"),
    ]
    client = FakeClient([
        _response(tool_calls=[
            _tool_call("1", "open_site", '{"name": "youtube"}'),
            _tool_call("2", "open_site", '{"name": "instagram"}'),
        ])
    ])
    out = _specialist(client, tools).run("open youtube and instagram")
    assert out == "opened youtube; opened instagram"
    assert client.calls == 1


def test_no_tool_returns_text_answer():
    client = FakeClient([_response(content="I can't do that in the browser.", tool_calls=None)])
    out = _specialist(client, []).run("hmm")
    assert out == "I can't do that in the browser."


def test_unknown_tool_is_handled():
    client = FakeClient([_response(tool_calls=[_tool_call("1", "nope", "{}")])])
    out = _specialist(client, []).run("do something")
    assert out.startswith("error: unknown tool")


def test_tool_exception_becomes_error_string():
    def boom():
        raise RuntimeError("chrome exploded")

    tools = [ToolSpec("open_shorts", "shorts", no_params(), boom)]
    client = FakeClient([_response(tool_calls=[_tool_call("1", "open_shorts", "{}")])])
    out = _specialist(client, tools).run("open shorts")
    assert out == "error: chrome exploded"  # loop survives a tool failure


def test_bad_json_arguments_dont_crash():
    tools = [ToolSpec("open_shorts", "shorts", no_params(), lambda: "ok")]
    client = FakeClient([_response(tool_calls=[_tool_call("1", "open_shorts", "not json")])])
    # empty kwargs -> callable takes none -> succeeds
    assert _specialist(client, tools).run("x") == "ok"


def test_chain_mode_loops_until_no_tool_calls():
    order = []
    tools = [
        ToolSpec("step", "a step", no_params(), lambda: order.append("ran") or "did a step"),
    ]
    client = FakeClient([
        _response(tool_calls=[_tool_call("1", "step", "{}")]),   # round 1: call tool
        _response(content="all done", tool_calls=None),           # round 2: finish
    ])
    out = _specialist(client, tools, chain=True).run("go")
    assert out == "all done"
    assert order == ["ran"]
    assert client.calls == 2
