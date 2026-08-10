"""Pick the AI backend(s) to complete the assignment, in the user's preferred order.

Preference comes from config.ai_app_order (default 'claude_desktop,browser'). The
orchestrator tries each *available* backend in turn and uses the first that returns
an answer, so it gracefully falls back (e.g. Claude desktop → browser).
"""
from __future__ import annotations

import logging

from jarvis.ai_apps.base import AnswerBackend
from jarvis.ai_apps.browser_backend import BrowserBackend
from jarvis.ai_apps.claude_desktop import ClaudeDesktopBackend
from jarvis.config import config

logger = logging.getLogger("jarvis.ai")

_ALL = {
    "claude_desktop": ClaudeDesktopBackend,
    "browser": BrowserBackend,
}


def available_backends(order=None) -> list[AnswerBackend]:
    """Available backends in preferred order (first = most preferred)."""
    order = order or config.ai_app_order
    out: list[AnswerBackend] = []
    for name in order:
        cls = _ALL.get(name)
        if not cls:
            continue
        b = cls()
        if b.available():
            out.append(b)
    logger.info("AI backends available: %s", [b.name for b in out])
    return out
