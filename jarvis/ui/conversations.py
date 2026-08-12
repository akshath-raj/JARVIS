"""Persistent conversation history for the dashboard.

The long-term MemoryStore keeps only a *distilled* profile + a transient log that
gets consumed by summarization, so it can't back a "show my past conversations"
view. This is a separate, append-only transcript the HUD reads. Turns are grouped
into sessions (a fresh session id per assistant boot) so the UI can list them.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class ConversationLog:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session = int(time.time())
        self._lock = threading.Lock()

    def add(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text or role not in ("user", "assistant"):
            return
        rec = {"ts": time.time(), "session": self._session, "role": role, "text": text}
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _read(self) -> list[dict]:
        try:
            with self._path.open(encoding="utf-8") as f:
                out = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return out
        except OSError:
            return []

    def sessions(self, limit: int = 20) -> list[dict]:
        """Return recent conversation sessions, newest first, each as
        {id, started, title, turns:[{role, text, ts}]}."""
        rows = self._read()
        by_id: dict[int, list[dict]] = {}
        for r in rows:
            by_id.setdefault(r.get("session", 0), []).append(r)
        sessions = []
        for sid, turns in by_id.items():
            turns.sort(key=lambda t: t.get("ts", 0))
            first_user = next((t["text"] for t in turns if t.get("role") == "user"), "")
            sessions.append(
                {
                    "id": sid,
                    "started": turns[0].get("ts", sid) if turns else sid,
                    "title": (first_user or "Conversation")[:80],
                    "turns": [
                        {"role": t.get("role"), "text": t.get("text", ""), "ts": t.get("ts")}
                        for t in turns
                    ],
                }
            )
        sessions.sort(key=lambda s: s["started"], reverse=True)
        return sessions[:limit]
