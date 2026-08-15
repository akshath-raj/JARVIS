"""Persisted user identity — the name JARVIS greets you by and uses everywhere.

Stored at ~/.jarvis/user.json so it survives across sessions. On the very first HUD
launch, when no name is stored (and JARVIS_UI_USER isn't set), the dashboard asks
for it once and saves it here; every later session just loads it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_PATH = Path(os.path.expanduser(os.getenv("JARVIS_USER_FILE", "~/.jarvis/user.json")))


def load_name() -> str:
    """Return the saved name, or "" if none has been set yet."""
    try:
        return (json.loads(_PATH.read_text()).get("name") or "").strip()
    except Exception:
        return ""


def save_name(name: str) -> str:
    """Persist the name (trimmed, capped). Returns the stored value ("" if blank)."""
    name = " ".join((name or "").split())[:40]
    if not name:
        return ""
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps({"name": name}))
    except Exception:
        pass
    return name
