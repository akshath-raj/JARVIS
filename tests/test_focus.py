"""Tests for focus-assist mode: classification, closing/blocking distractions,
the timed regime, and the browser-tool guard. Offline — a fake `closer` stands in
for the AppleScript that shuts Chrome tabs, and the timer's "minute" is shrunk."""
from __future__ import annotations

import asyncio

from jarvis.focus.controller import FocusController, _pretty, resolve_technique


class FakeCloser:
    """Simulates open distraction tabs; closes those matching the blocklist."""
    def __init__(self, open_hosts=None):
        self.open = list(open_hosts or [])
        self.calls = 0

    def __call__(self, sites):
        self.calls += 1
        closed = [h for h in self.open if any(s in h for s in sites)]
        self.open = [h for h in self.open if h not in closed]
        return closed


# ── classification / helpers ─────────────────────────────────────────────────
def test_is_distraction():
    f = FocusController()
    assert f.is_distraction("instagram")
    assert f.is_distraction("https://www.youtube.com/watch?v=x")
    assert f.is_distraction("netflix.com")
    assert not f.is_distraction("github")
    assert not f.is_distraction("my notes.pdf")
    assert not f.is_distraction("")


def test_resolve_technique_aliases():
    assert resolve_technique("pomo") == "pomodoro"
    assert resolve_technique("ultradian") == "90-20"
    assert resolve_technique("flow") == "flowtime"
    assert resolve_technique("") == "pomodoro"
    assert resolve_technique("nonsense") == "pomodoro"


def test_pretty_dedupes_and_labels():
    assert _pretty(["youtube.com", "youtube.com", "instagram.com"]) == ["YouTube", "Instagram"]


# ── lifecycle: close on start, block on reopen, stop ─────────────────────────
def test_start_closes_current_distractions():
    async def run():
        closer = FakeCloser(["youtube.com", "instagram.com", "github.com"])
        f = FocusController(closer=closer, minute=0.001)
        msg = await f.start("pomodoro")
        assert f.active and "Focus mode engaged" in msg
        assert "youtube.com" not in closer.open and "instagram.com" not in closer.open
        assert "github.com" in closer.open   # not a distraction → left alone
        f.stop()
    asyncio.run(run())


def test_watch_closes_reopened_distraction_and_announces():
    async def run():
        closer = FakeCloser(["youtube.com"])
        said = []
        async def announce(t): said.append(t)
        f = FocusController(announce=announce, closer=closer, poll_seconds=0.02, minute=0.001)
        await f.start("pomodoro")
        assert closer.open == []                 # closed on start
        closer.open = ["netflix.com"]            # user sneaks a distraction back open
        await asyncio.sleep(0.15)
        assert "netflix.com" not in closer.open  # watcher shut it
        assert any("focus mode" in s.lower() for s in said)
        f.stop()
    asyncio.run(run())


def test_timer_runs_sprints_and_stop_reports():
    async def run():
        f = FocusController(closer=lambda s: [], poll_seconds=0.05, minute=0.002)
        await f.start("pomodoro")                # work 25 → ~0.05s per sprint
        await asyncio.sleep(0.25)
        st = f.status()
        assert st["active"] and st["cycles"] >= 1 and st["phase"] in ("work", "break")
        stop_msg = f.stop()
        assert not f.active and "Focus mode ended" in stop_msg
    asyncio.run(run())


def test_stop_when_inactive():
    f = FocusController()
    assert "isn't on" in f.stop()


def test_sweep_kills_lingering_profile_clone():
    """A second (browser-agent) clone Chrome blinds the AppleScript sweep, so focus
    quits it on every sweep — both on start and while watching for reopens."""
    async def run():
        killed = []
        closer = FakeCloser(["youtube.com"])
        f = FocusController(
            closer=closer, poll_seconds=0.02, minute=0.001,
            clone_dir="/Users/x/.jarvis/chrome-clone",
            clone_killer=lambda d: killed.append(d) or True,
        )
        await f.start("pomodoro")
        assert killed and killed[0] == "/Users/x/.jarvis/chrome-clone"
        before = len(killed)
        await asyncio.sleep(0.1)          # watch loop keeps quitting the clone
        assert len(killed) > before
        f.stop()
    asyncio.run(run())


def test_no_clone_dir_skips_kill():
    """With no clone dir configured, focus never tries to kill a clone."""
    async def run():
        killed = []
        f = FocusController(closer=lambda s: [], minute=0.001,
                            clone_killer=lambda d: killed.append(d) or True)
        await f.start("pomodoro")
        assert killed == []
        f.stop()
    asyncio.run(run())


def test_focus_pushes_to_ui_state():
    from jarvis.ui.controller import UIController

    async def run():
        ui = UIController(user="Akshath")
        f = FocusController(closer=lambda s: [], ui=ui, minute=0.001)
        await f.start("pomodoro")
        focus = ui.state()["focus"]
        assert focus["active"] and focus["technique"] == "pomodoro"
        assert "ends_at" in focus and "cycles" in focus
        assert ui.is_revealed()          # a focus session reveals the HUD
        f.stop()
        assert ui.state()["focus"] == {}  # cleared on stop
    asyncio.run(run())


# ── browser tool refuses distractions while focus is on ──────────────────────
async def _invoke(tool, **kwargs):
    import json

    from agents.tool_context import ToolContext

    args = json.dumps(kwargs)
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="t1", tool_arguments=args)
    return await tool.on_invoke_tool(ctx, args)


def test_browser_tools_block_distraction_when_focused():
    from unittest.mock import MagicMock

    from jarvis.openai_agent.tools import build_browser_tools

    async def run():
        f = FocusController(closer=lambda s: [], minute=0.001)
        browser = MagicMock()
        browser.open_site.return_value = "opened"
        tools = {t.name: t for t in
                 build_browser_tools(browser=browser, memory=MagicMock(), focus=f)}

        # not focused → distraction opens normally
        await _invoke(tools["open_site"], name="instagram")
        assert browser.open_site.called

        # focused → the same request is refused, and the browser is NOT driven
        browser.reset_mock()
        await f.start("pomodoro")
        res = await _invoke(tools["open_site"], name="instagram")
        assert "focus mode" in res.lower() and "blocked" in res.lower()
        browser.open_site.assert_not_called()

        # a non-distraction still opens while focused
        await _invoke(tools["open_site"], name="github")
        assert browser.open_site.called
        f.stop()

    asyncio.run(run())
