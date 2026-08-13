"""Focus assist mode: close distractions, block reopening them, and run a timed
focus regime (Pomodoro and other techniques) until the user ends it."""
from __future__ import annotations

from jarvis.focus.controller import TECHNIQUES, FocusController

__all__ = ["FocusController", "TECHNIQUES"]
