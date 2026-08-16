"""Persistent store for to-dos, reminders, and calendar events.

One JSON file (``tasks.json``) under the memory dir, guarded by a lock. Fuzzy
matching by normalised substring lets the user say "I'm done with the assignment"
and have JARVIS find the right item. Everything is local.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _id() -> str:
    return uuid.uuid4().hex[:8]


class TaskStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = {"todos": [], "reminders": [], "events": []}
        self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            for k in ("todos", "reminders", "events"):
                self._data.setdefault(k, [])
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _match(items: list[dict], query: str, field: str):
        q = _norm(query)
        if not q:
            return None
        # prefer exact, then substring either way
        for it in items:
            if _norm(it.get(field, "")) == q:
                return it
        for it in items:
            v = _norm(it.get(field, ""))
            if q in v or v in q:
                return it
        return None

    # ── to-dos ──────────────────────────────────────────────────────────────
    def add_todo(self, text: str) -> dict:
        todo = {"id": _id(), "text": text.strip(), "done": False, "created": time.time()}
        with self._lock:
            self._data["todos"].append(todo)
            self._save()
        return todo

    def list_todos(self, include_done: bool = False) -> list[dict]:
        with self._lock:
            return [t for t in self._data["todos"] if include_done or not t.get("done")]

    def complete_todo(self, query: str) -> dict | None:
        with self._lock:
            todo = self._match([t for t in self._data["todos"] if not t.get("done")], query, "text")
            if todo:
                todo["done"] = True
                todo["done_at"] = time.time()
                self._save()
            return todo

    def remove_todo(self, query: str) -> dict | None:
        with self._lock:
            todo = self._match(self._data["todos"], query, "text")
            if todo:
                self._data["todos"].remove(todo)
                self._save()
            return todo

    def remove_todo_by_id(self, tid: str) -> dict | None:
        """Delete a to-do by its exact id (the HUD delete button passes it back)."""
        with self._lock:
            for t in self._data["todos"]:
                if t.get("id") == tid:
                    self._data["todos"].remove(t)
                    self._save()
                    return t
            return None

    # ── reminders ─────────────────────────────────────────────────────────────
    def add_reminder(self, text: str, due_ts: float) -> dict:
        rem = {
            "id": _id(), "text": text.strip(), "due": float(due_ts),
            "fired": False, "cancelled": False, "created": time.time(),
        }
        with self._lock:
            self._data["reminders"].append(rem)
            self._save()
        return rem

    def list_reminders(self, pending_only: bool = True) -> list[dict]:
        with self._lock:
            rs = self._data["reminders"]
            out = [r for r in rs if not r.get("cancelled") and (not pending_only or not r.get("fired"))]
            return sorted(out, key=lambda r: r.get("due", 0))

    def due_reminders(self, now: float) -> list[dict]:
        with self._lock:
            return [
                r for r in self._data["reminders"]
                if not r.get("fired") and not r.get("cancelled") and r.get("due", 0) <= now
            ]

    def mark_fired(self, rid: str) -> None:
        with self._lock:
            for r in self._data["reminders"]:
                if r["id"] == rid:
                    r["fired"] = True
            self._save()

    def cancel_reminder(self, query: str) -> dict | None:
        with self._lock:
            rem = self._match(
                [r for r in self._data["reminders"] if not r.get("cancelled") and not r.get("fired")],
                query, "text",
            )
            if rem:
                rem["cancelled"] = True
                self._save()
            return rem

    def cancel_reminder_by_id(self, rid: str) -> dict | None:
        """Cancel a pending reminder by its exact id (the HUD agenda ⌫ button)."""
        with self._lock:
            for r in self._data["reminders"]:
                if r.get("id") == rid and not r.get("cancelled"):
                    r["cancelled"] = True
                    self._save()
                    return r
            return None

    # ── calendar events ───────────────────────────────────────────────────────
    def add_event(self, title: str, start_ts: float, end_ts: float | None = None) -> dict:
        ev = {
            "id": _id(), "title": title.strip(), "start": float(start_ts),
            "end": float(end_ts) if end_ts else None, "created": time.time(),
        }
        with self._lock:
            self._data["events"].append(ev)
            self._save()
        return ev

    def list_events(self, upcoming_only: bool = True) -> list[dict]:
        now = time.time()
        with self._lock:
            out = [e for e in self._data["events"] if not upcoming_only or e.get("start", 0) >= now - 3600]
            return sorted(out, key=lambda e: e.get("start", 0))

    def remove_event(self, query: str) -> dict | None:
        with self._lock:
            ev = self._match(self._data["events"], query, "title")
            if ev:
                self._data["events"].remove(ev)
                self._save()
            return ev

    def remove_event_by_id(self, eid: str) -> dict | None:
        """Delete a calendar event by its exact id (the HUD agenda ⌫ button)."""
        with self._lock:
            for ev in self._data["events"]:
                if ev.get("id") == eid:
                    self._data["events"].remove(ev)
                    self._save()
                    return ev
            return None
