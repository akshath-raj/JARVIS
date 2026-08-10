"""Announcer — lets a background browser task speak its result later.

Browser tasks run for 30s–2min in the background, so `browser_task` returns a
quick spoken ack and the result is delivered when it's ready. The Announcer holds
the live AgentSession (set by the entrypoint after the session starts) and speaks
via TTS. `announce(text)` is a no-op until the session is attached.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("jarvis.browser")


class Announcer:
    def __init__(self) -> None:
        self.session = None

    async def announce(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.session is None:
            logger.info("(no session to announce) %s", text)
            return
        try:
            await self.session.say(text)
        except Exception as e:  # never let a background announcement crash anything
            logger.warning("announce failed: %s", e)
