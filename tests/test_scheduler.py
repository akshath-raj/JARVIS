"""Tests for the calendar / to-do / reminder + alarm subsystem."""
from __future__ import annotations

import asyncio
import datetime
import time

from jarvis.scheduler import AlarmScheduler, TaskStore, parse_when


# ── time parsing ─────────────────────────────────────────────────────────────
def test_parse_when_absolute_and_relative():
    base = datetime.datetime(2026, 8, 12, 18, 0, 0)
    assert parse_when("2026-08-12T21:00:00") == datetime.datetime(2026, 8, 12, 21, 0, 0)
    assert parse_when("in 1 hour", base) == datetime.datetime(2026, 8, 12, 19, 0, 0)
    assert parse_when("at 9pm today", base) == datetime.datetime(2026, 8, 12, 21, 0, 0)
    assert parse_when("in another hour", base) == datetime.datetime(2026, 8, 12, 19, 0, 0)
    assert parse_when("gibberish not a time xyz") is None


# ── store CRUD ───────────────────────────────────────────────────────────────
def test_todo_lifecycle(tmp_path):
    s = TaskStore(str(tmp_path / "t.json"))
    s.add_todo("buy milk")
    s.add_todo("submit the assignment")
    assert {t["text"] for t in s.list_todos()} == {"buy milk", "submit the assignment"}
    done = s.complete_todo("assignment")          # fuzzy substring match
    assert done["text"] == "submit the assignment"
    assert [t["text"] for t in s.list_todos()] == ["buy milk"]


def test_remove_by_id(tmp_path):
    # the HUD delete buttons pass the item's exact id back so the right row is
    # removed even when two items share the same text.
    s = TaskStore(str(tmp_path / "t.json"))
    a = s.add_todo("call the bank")
    b = s.add_todo("call the bank")            # duplicate text, distinct id
    assert s.remove_todo_by_id(a["id"])["id"] == a["id"]
    assert [t["id"] for t in s.list_todos()] == [b["id"]]
    assert s.remove_todo_by_id("nope") is None

    now = time.time()
    r = s.add_reminder("standup", now + 3600)
    assert s.cancel_reminder_by_id(r["id"])["id"] == r["id"]
    assert s.list_reminders() == []            # cancelled → no longer pending
    assert s.cancel_reminder_by_id(r["id"]) is None  # already cancelled

    e = s.add_event("dentist", now + 7200)
    assert s.remove_event_by_id(e["id"])["id"] == e["id"]
    assert s.list_events() == []
    assert s.remove_event_by_id("ghost") is None


def test_store_persists_across_instances(tmp_path):
    p = str(tmp_path / "t.json")
    TaskStore(p).add_todo("persist me")
    assert [t["text"] for t in TaskStore(p).list_todos()] == ["persist me"]


def test_reminders_and_events(tmp_path):
    s = TaskStore(str(tmp_path / "t.json"))
    now = time.time()
    s.add_reminder("ring soon", now - 1)          # already due
    s.add_reminder("later", now + 3600)
    assert {r["text"] for r in s.due_reminders(now)} == {"ring soon"}
    s.add_event("dentist", now + 7200)
    assert [e["title"] for e in s.list_events()] == ["dentist"]


# ── scheduler add_reminder (parsing + linked to-do) ──────────────────────────
def test_add_reminder_creates_linked_todo(tmp_path):
    s = TaskStore(str(tmp_path / "t.json"))
    sched = AlarmScheduler(s)
    msg = sched.add_reminder("submit the assignment", "in 2 hours")
    assert "Reminder set" in msg
    # surfaced on the to-do list so "I'm done" can clear it
    assert any("assignment" in t["text"] for t in s.list_todos())


def test_add_reminder_rejects_past(tmp_path):
    s = TaskStore(str(tmp_path / "t.json"))
    sched = AlarmScheduler(s)
    assert "already passed" in sched.add_reminder("x", "2020-01-01T00:00:00")


# ── firing + dismissal ───────────────────────────────────────────────────────
class _FakePlayer:
    def __init__(self):
        self.ringing = False
        self.log = []

    def start(self):
        self.ringing = True
        self.log.append("start")

    def stop(self):
        self.ringing = False
        self.log.append("stop")


class _FakeUI:
    def __init__(self):
        self.calls = []

    def show_alarm(self, a):
        self.calls.append(("show", a["text"]))

    def clear_alarm(self, i=None):
        self.calls.append(("clear", i))


def test_alarm_fires_and_dismisses(tmp_path):
    s = TaskStore(str(tmp_path / "t.json"))
    s.add_reminder("submit the assignment", time.time() - 1)  # due now
    ui = _FakeUI()
    spoken = []

    async def announce(t):
        spoken.append(t)

    sched = AlarmScheduler(s, announce=announce, ui=ui)
    sched._player = _FakePlayer()

    async def run():
        sched.start()
        # first tick fires immediately (due in the past); poll briefly
        for _ in range(30):
            if sched.active_alarms:
                break
            await asyncio.sleep(0.2)
        assert [a["text"] for a in sched.active_alarms] == ["submit the assignment"]
        assert sched._player.log == ["start"]
        assert ui.calls[0] == ("show", "submit the assignment")
        assert spoken and "submit the assignment" in spoken[0]
        # dismiss (STOP button / voice)
        d = sched.dismiss()
        assert d["text"] == "submit the assignment"
        assert sched._player.log[-1] == "stop"
        assert not sched.active_alarms
        assert s.list_reminders() == []  # fired → no longer pending

    asyncio.run(run())


# ── planner tools wired ──────────────────────────────────────────────────────
def test_build_planner_tools_present(tmp_path):
    from jarvis.openai_agent.tools import build_planner_tools

    sched = AlarmScheduler(TaskStore(str(tmp_path / "t.json")))
    names = {t.name for t in build_planner_tools(scheduler=sched)}
    assert {"add_reminder", "add_todo", "complete_todo", "add_calendar_event",
            "stop_alarm", "current_time", "list_reminders"} <= names


def test_ui_state_includes_tasks_and_alarms(tmp_path):
    from jarvis.ui import UIController

    s = TaskStore(str(tmp_path / "t.json"))
    s.add_todo("do homework")
    s.add_event("dentist", time.time() + 3600)
    ui = UIController(user="A", memory=None, conversations=None, tasks=s)
    ui.show_alarm({"id": "1", "text": "ring", "due": time.time()})
    st = ui.state(0)
    assert [t["text"] for t in st["todos"]] == ["do homework"]
    assert any(a["title"] == "dentist" for a in st["agenda"])
    assert st["alarms"][0]["text"] == "ring"
    ui.clear_alarm("1")
    assert ui.state(0)["alarms"] == []
