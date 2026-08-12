"""UIController — the bridge between the voice agent and the HUD dashboard.

The agent (running in its own asyncio loop / worker threads) calls the `reveal`,
`hide`, `show_explanation`, `navigate` methods. The web server (running in a
separate uvicorn thread) calls `state(since=...)` on each poll. Both sides only
touch this object, guarded by a lock, so there's no cross-event-loop coupling —
the browser polls and applies whatever events have accumulated.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class UIController:
    def __init__(self, *, user: str, memory=None, conversations=None, tasks=None) -> None:
        self._user = user
        self._memory = memory
        self._conversations = conversations
        self._tasks = tasks  # TaskStore: to-dos, calendar events, reminders
        self._lock = threading.Lock()
        self._events: deque[dict] = deque(maxlen=200)
        self._seq = 0
        self._revealed = False
        # the most recent explanation, so a client that connects late still sees it
        self._last_explanation: Optional[dict] = None
        self._alarms: dict = {}  # id -> currently ringing alarm (shown as a popup)
        # optional push hook — the WebSocket server registers this to deliver an
        # event to connected browsers the instant it's emitted (no polling).
        self._notifier = None
        # optional inbound-command hook — the server calls this with a dict when the
        # browser sends a command (e.g. the STOP-alarm button).
        self._command_handler = None

    def set_notifier(self, callback) -> None:
        """Register a thread-safe callback(event: dict) fired on each new event."""
        self._notifier = callback

    def set_command_handler(self, callback) -> None:
        """Register a callback(command: dict) for messages the browser sends back
        (e.g. {'cmd': 'dismiss_alarm', 'id': ...})."""
        self._command_handler = callback

    def handle_command(self, command: dict) -> None:
        if self._command_handler is not None:
            try:
                self._command_handler(command)
            except Exception:
                pass

    # ── event emitters (called from the agent) ──────────────────────────────
    def _emit(self, event: dict) -> int:
        with self._lock:
            self._seq += 1
            full = {**event, "id": self._seq, "ts": time.time()}
            self._events.append(full)
            seq = self._seq
        # notify outside the lock so a slow transport can't stall the agent
        if self._notifier is not None:
            try:
                self._notifier(full)
            except Exception:
                pass
        return seq

    def reveal(self) -> None:
        with self._lock:
            self._revealed = True
        self._emit({"type": "reveal"})

    def hide(self) -> None:
        with self._lock:
            self._revealed = False
        self._emit({"type": "hide"})

    def show_explanation(
        self, *, title: str, body: str, image_b64: str = "", source: str = ""
    ) -> None:
        """Render an explanation/analysis in the main HUD panel (the key feature —
        e.g. a screenshot's detailed GPT reading). Reveals the HUD if hidden."""
        expl = {
            "type": "explanation",
            "title": title or "Analysis",
            "body": body or "",
            "image": image_b64 or "",
            "source": source or "",
        }
        with self._lock:
            self._revealed = True
            self._last_explanation = expl
        self._emit(expl)

    def navigate(self, section: str, item_id=None) -> None:
        """Direct the user to a section (about / memories / conversations) and,
        optionally, a specific item."""
        with self._lock:
            self._revealed = True
        self._emit({"type": "navigate", "section": section, "item": item_id})

    def show_alarm(self, alarm: dict) -> None:
        """Pop up a ringing-alarm overlay on the HUD with a STOP button."""
        with self._lock:
            self._revealed = True
            self._alarms[alarm["id"]] = alarm
        self._emit({"type": "alarm", "alarm": alarm})

    def clear_alarm(self, alarm_id=None) -> None:
        """Dismiss the alarm popup (one, or all when id is None)."""
        with self._lock:
            if alarm_id is None:
                self._alarms.clear()
            else:
                self._alarms.pop(alarm_id, None)
        self._emit({"type": "alarm_cleared", "id": alarm_id})

    def is_revealed(self) -> bool:
        with self._lock:
            return self._revealed

    # ── data + polling (called from the web server) ─────────────────────────
    def _about(self) -> list[str]:
        if self._memory is None:
            return []
        try:
            return self._memory.all()
        except Exception:
            return []

    def _sessions(self) -> list[dict]:
        if self._conversations is None:
            return []
        try:
            return self._conversations.sessions()
        except Exception:
            return []

    def _tasks_data(self) -> dict:
        if self._tasks is None:
            return {"todos": [], "agenda": []}
        try:
            todos = self._tasks.list_todos()
        except Exception:
            todos = []
        try:
            events = [{"kind": "event", **e} for e in self._tasks.list_events()]
            reminders = [
                {"kind": "reminder", "title": r["text"], "start": r["due"], **r}
                for r in self._tasks.list_reminders()
            ]
            agenda = sorted(events + reminders, key=lambda x: x.get("start", 0))
        except Exception:
            agenda = []
        return {"todos": todos, "agenda": agenda}

    def state(self, since: int = 0) -> dict:
        with self._lock:
            revealed = self._revealed
            events = [e for e in self._events if e["id"] > since]
            cursor = self._seq
            last_expl = self._last_explanation
            alarms = list(self._alarms.values())
        about = self._about()
        tasks = self._tasks_data()
        return {
            "user": self._user,
            "revealed": revealed,
            "cursor": cursor,
            "about": about,
            "memories": about,  # profile bullets double as the memory list
            "conversations": self._sessions(),
            "todos": tasks["todos"],
            "agenda": tasks["agenda"],
            "alarms": alarms,
            "last_explanation": last_expl,
            "events": events,
        }
