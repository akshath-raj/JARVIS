"""Local macOS system actions (no cloud): Spotlight file search + Calendar read."""
from __future__ import annotations

import subprocess


class SystemError_(RuntimeError):
    pass


def _run(args: list[str], timeout: int = 20) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SystemError_(proc.stderr.strip() or "command failed")
    return proc.stdout.strip()


# ── Files (Spotlight) ────────────────────────────────────────────────────
def spotlight_find(query: str, limit: int = 5) -> list[str]:
    """Find files by name/content via Spotlight (mdfind)."""
    out = _run(["mdfind", "-name", query])
    hits = [line for line in out.splitlines() if line][:limit]
    return hits


def open_path(path: str) -> None:
    """Open a file, folder, or app with the default handler."""
    _run(["open", path], timeout=10)


# ── Calendar (Calendar.app via AppleScript) ───────────────────────────────
_TODAY_EVENTS = r'''
set _now to current date
set _startOfDay to _now - (time of _now)
set _endOfDay to _startOfDay + (1 * days)
set _out to ""
tell application "Calendar"
  repeat with _cal in calendars
    repeat with _ev in (every event of _cal whose start date is greater than or equal to _startOfDay and start date is less than _endOfDay)
      set _out to _out & (summary of _ev) & " at " & (time string of (start date of _ev)) & linefeed
    end repeat
  end repeat
end tell
return _out
'''


def calendar_today() -> list[str]:
    """Return today's calendar events as 'Title at H:MM' strings."""
    out = _run(["osascript", "-e", _TODAY_EVENTS], timeout=30)
    return [line for line in out.splitlines() if line.strip()]
