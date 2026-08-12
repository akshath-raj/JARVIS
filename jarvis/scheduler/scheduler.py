"""AlarmScheduler — the brain of reminders/alarms.

Owns the TaskStore and the AlarmPlayer, and runs a lightweight async tick that
fires reminders when their time comes: it starts the ringing alarm, speaks the
reminder aloud (via the Announcer), and shows a dismissable popup on the HUD.

Dismissal (voice "stop the alarm" / "I'm done", or the HUD STOP button) is
`dismiss()` — thread-safe and loop-free, so it works from either the voice loop
or the web-server thread.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import threading

from jarvis.scheduler.alarm import AlarmPlayer
from jarvis.scheduler.store import TaskStore
from jarvis.scheduler.timeparse import parse_when

logger = logging.getLogger("jarvis.scheduler")


def _fmt(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    today = datetime.date.today()
    day = "" if dt.date() == today else (
        " tomorrow" if dt.date() == today + datetime.timedelta(days=1) else dt.strftime(" on %a %d %b")
    )
    return dt.strftime("%-I:%M %p").lower() + day


class AlarmScheduler:
    def __init__(self, store: TaskStore, *, announce=None, ui=None, sound=None) -> None:
        self.store = store
        self._announce = announce           # async callable(text)
        self._ui = ui                        # UIController (optional)
        self._player = AlarmPlayer(sound) if sound else AlarmPlayer()
        self._active: dict[str, dict] = {}   # id -> reminder currently ringing
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the tick on the current (LiveKit) event loop."""
        try:
            self._loop = asyncio.get_running_loop()
            self._task = self._loop.create_task(self._run())
        except RuntimeError:
            logger.warning("no running loop; scheduler tick not started")

    async def _run(self) -> None:
        while True:
            try:
                import time as _t
                for rem in self.store.due_reminders(_t.time()):
                    await self._fire(rem)
            except Exception as e:
                logger.warning("scheduler tick error: %s", e)
            await asyncio.sleep(5)

    async def _fire(self, rem: dict) -> None:
        self.store.mark_fired(rem["id"])
        with self._lock:
            self._active[rem["id"]] = rem
        self._player.start()
        if self._ui is not None:
            try:
                self._ui.show_alarm({"id": rem["id"], "text": rem["text"], "due": rem.get("due")})
            except Exception:
                pass
        if self._announce is not None:
            try:
                await self._announce(f"Sir, this is your reminder: {rem['text']}.")
            except Exception:
                pass
        logger.info("alarm fired: %s", rem["text"])

    # ── dismissal (loop-free, any thread) ────────────────────────────────────
    def dismiss(self, rid: str | None = None) -> dict | None:
        with self._lock:
            if rid is None:
                dismissed = next(iter(self._active.values()), None)
                self._active.clear()
            else:
                dismissed = self._active.pop(rid, None)
            still_ringing = bool(self._active)
        if not still_ringing:
            self._player.stop()
        if self._ui is not None:
            try:
                self._ui.clear_alarm(rid)
            except Exception:
                pass
        return dismissed

    @property
    def active_alarms(self) -> list[dict]:
        with self._lock:
            return list(self._active.values())

    # ── tool-facing operations ────────────────────────────────────────────────
    def add_reminder(self, text: str, when: str) -> str:
        due = parse_when(when)
        if due is None:
            return f"I couldn't work out when '{when}' is, sir — try 'at 9pm' or 'in 1 hour'."
        import time as _t
        if due.timestamp() <= _t.time():
            return "That time has already passed, sir."
        self.store.add_reminder(text, due.timestamp())
        # a reminder is a task the user needs to do — surface it on the to-do list
        # too (deduped) so "I'm done with X" can clear it.
        if not any(text.lower() in t["text"].lower() or t["text"].lower() in text.lower()
                   for t in self.store.list_todos()):
            self.store.add_todo(text)
        return f"Reminder set: {text} at {_fmt(due.timestamp())}."

    def add_event(self, title: str, when: str) -> str:
        start = parse_when(when)
        if start is None:
            return f"I couldn't work out when '{when}' is, sir."
        self.store.add_event(title, start.timestamp())
        return f"Added to your calendar: {title} at {_fmt(start.timestamp())}."
