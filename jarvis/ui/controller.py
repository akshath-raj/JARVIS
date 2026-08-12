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
    def __init__(self, *, user: str, memory=None, conversations=None) -> None:
        self._user = user
        self._memory = memory
        self._conversations = conversations
        self._lock = threading.Lock()
        self._events: deque[dict] = deque(maxlen=200)
        self._seq = 0
        self._revealed = False
        # the most recent explanation, so a client that connects late still sees it
        self._last_explanation: Optional[dict] = None
        # optional push hook — the WebSocket server registers this to deliver an
        # event to connected browsers the instant it's emitted (no polling).
        self._notifier = None

    def set_notifier(self, callback) -> None:
        """Register a thread-safe callback(event: dict) fired on each new event."""
        self._notifier = callback

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

    def state(self, since: int = 0) -> dict:
        with self._lock:
            revealed = self._revealed
            events = [e for e in self._events if e["id"] > since]
            cursor = self._seq
            last_expl = self._last_explanation
        about = self._about()
        return {
            "user": self._user,
            "revealed": revealed,
            "cursor": cursor,
            "about": about,
            "memories": about,  # profile bullets double as the memory list
            "conversations": self._sessions(),
            "last_explanation": last_expl,
            "events": events,
        }
