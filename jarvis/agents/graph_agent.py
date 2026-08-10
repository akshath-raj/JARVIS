"""Thin voice agent for the LangGraph brain.

In LangGraph mode the reasoning + tools live entirely in the compiled graph
(wrapped by VoiceLLMAdapter as the session LLM). This agent therefore carries no
tools — it only keeps the two things that must live at the LiveKit layer: the
wake-word gate (via BaseJarvisAgent) and the spoken greeting.
"""
from __future__ import annotations

from jarvis.agents.base import BaseJarvisAgent

GREETING = "Good day, sir. JARVIS, at your service."


class GraphAgent(BaseJarvisAgent):
    def __init__(self) -> None:
        # Instructions are minimal: the graph supplies the real system prompt
        # (persona + routing + injected memories) on every turn.
        super().__init__(instructions="You are JARVIS, a concise British butler.")

    async def on_enter(self) -> None:
        if not self.session.userdata.greeted:
            self.session.userdata.greeted = True
            # A fixed spoken line (not an LLM turn) — fast and deterministic, and a
            # statement (not a question) so the session starts asleep.
            await self.session.say(GREETING)
