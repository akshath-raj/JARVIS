"""Shared persona + activation behaviour for every JARVIS agent.

`BaseJarvisAgent` enforces wake-word activation with smart follow-up (see
jarvis.activation) uniformly across all agents, including after a handoff.
"""
from __future__ import annotations

import logging

from livekit.agents import Agent, StopResponse, llm

logger = logging.getLogger("jarvis.agent")

# Appended to every agent's instructions. Keeps replies speakable and disables
# qwen3's chain-of-thought for low latency (harmless for non-thinking models).
VOICE_STYLE = (
    "You are speaking out loud, so keep replies short and natural: a sentence or "
    "two, no markdown, no lists, no emoji. You are JARVIS, a witty, unflappable "
    "British AI butler. Address the user as \"sir\" occasionally, never in every "
    "sentence. When you need more information, ask a brief clarifying QUESTION "
    "(end it with '?'). /no_think"
)


class BaseJarvisAgent(Agent):
    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Wake-word gate: answer only if woken or inside a follow-up window."""
        ctl = getattr(self.session.userdata, "activation", None)
        if ctl is None:
            return
        allowed, rewritten = ctl.gate(new_message.text_content or "")
        if not allowed:
            logger.info("Ignoring (not woken): %r", new_message.text_content)
            raise StopResponse()
        if rewritten is not None:
            # Strip the wake word so the agent sees just the request.
            new_message.content = [rewritten]
