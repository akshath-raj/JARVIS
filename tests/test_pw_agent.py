"""Tests for the Playwright-MCP autonomous web agent (offline — no real browser).

Exercises arg-building, the enabled/disabled + single-flight guards, and that
`browser_task` routes to the Playwright agent when it's enabled. `config` is a
frozen dataclass, so tests swap the module-level `config` reference for a stand-in.
"""
from __future__ import annotations

import asyncio
import types

from jarvis.browser_agent import pw_agent


def _cfg(**over):
    base = dict(pw_agent_enabled=True, pw_headless=False, pw_browser_channel="chrome",
                pw_user_data_dir="/tmp/pw", pw_viewport="1280x800", pw_caps="")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_mcp_args_include_channel_profile_and_headed(monkeypatch):
    monkeypatch.setattr(pw_agent, "config", _cfg())
    args = pw_agent._mcp_args()
    assert "@playwright/mcp@latest" in args
    assert args[args.index("--browser") + 1] == "chrome"
    assert "--user-data-dir" in args and "--viewport-size" in args
    assert "--headless" not in args           # headed by default so the user can watch
    assert "--caps" not in args               # none configured


def test_mcp_args_headless_and_caps(monkeypatch):
    monkeypatch.setattr(pw_agent, "config", _cfg(pw_headless=True, pw_caps="vision,pdf"))
    args = pw_agent._mcp_args()
    assert "--headless" in args
    assert args[args.index("--caps") + 1] == "vision,pdf"


def test_mcp_spec_defaults_to_playwright(monkeypatch):
    monkeypatch.setattr(pw_agent, "config", _cfg())     # no backend set → default
    name, args, instr, backend = pw_agent._mcp_spec()
    assert backend == "playwright" and name == "playwright"
    assert "@playwright/mcp@latest" in args
    assert "accessibility" in instr.lower()             # the Playwright playbook


def test_mcp_spec_selects_brocogni(monkeypatch):
    monkeypatch.setattr(
        pw_agent, "config",
        _cfg(browser_mcp_backend="brocogni", brocogni_args="--foo bar"),
    )
    name, args, instr, backend = pw_agent._mcp_spec()
    assert backend == "brocogni" and name == "brocogni"
    assert "browser-cognition-mcp" in args
    assert args[-2:] == ["--foo", "bar"]                # extra args appended
    assert "browser_observe" in instr                   # the brocogni playbook
    assert "@playwright/mcp@latest" not in args


def test_disabled_agent_returns_message(monkeypatch):
    monkeypatch.setattr(pw_agent, "config", _cfg(pw_agent_enabled=False))
    assert "turned off" in asyncio.run(pw_agent.run_web_task("do something"))


def test_report_single_flight(monkeypatch):
    """A second task while one is running is politely deferred (no overlap)."""
    monkeypatch.setattr(pw_agent, "config", _cfg())

    async def run():
        await pw_agent._lock.acquire()
        try:
            return await pw_agent.run_web_task_and_report("another task")
        finally:
            pw_agent._lock.release()

    assert "still on the last" in asyncio.run(run())


def test_report_reserves_browser_before_background_task_starts(monkeypatch):
    """Two immediate calls cannot both pass the pre-start lock check."""
    monkeypatch.setattr(pw_agent, "config", _cfg())

    async def fake_run(task):
        await asyncio.sleep(0.01)
        return "done"

    async def run():
        pw_agent._report_task = None
        monkeypatch.setattr(pw_agent, "run_web_task", fake_run)
        first = await pw_agent.run_web_task_and_report("first")
        second = await pw_agent.run_web_task_and_report("second")
        await pw_agent._report_task
        return first, second

    first, second = asyncio.run(run())
    assert "On it" in first
    assert "still on the last" in second


def test_browser_task_routes_to_pw_agent(monkeypatch):
    """When the Playwright agent is enabled, browser_task uses it (not browser-use)."""
    from jarvis.openai_agent import tools as T

    called = {}

    async def fake_pw(instruction, announce=None):
        called["pw"] = instruction
        return "On it, sir."

    async def fake_bu(instruction, announce=None):
        called["bu"] = instruction
        return "browser-use"

    monkeypatch.setattr(T, "_config", types.SimpleNamespace(pw_agent_enabled=True))
    monkeypatch.setattr(T, "run_web_task_and_report", fake_pw)
    monkeypatch.setattr(T, "run_browser_task", fake_bu)

    class _Mem:
        def log_activity(self, *a, **k):
            pass

    tools = {t.name: t for t in T.build_browser_tools(browser=object(), memory=_Mem())}
    bt = tools["browser_task"]

    import json

    from agents.tool_context import ToolContext

    args = json.dumps({"instruction": "search amazon for an ssd and open the first result"})
    ctx = ToolContext(context=None, tool_name="browser_task", tool_call_id="t1", tool_arguments=args)
    res = asyncio.run(bt.on_invoke_tool(ctx, args))
    assert "pw" in called and "bu" not in called
    assert res == "On it, sir."
