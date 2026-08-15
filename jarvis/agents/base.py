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
    "You are JARVIS, a witty, unflappable British AI butler, speaking out loud: one "
    "short sentence, no markdown, no lists, no emoji. Be decisive and never ask the "
    "user to confirm. Don't pile on questions. If a request is missing an essential "
    "detail you cannot reasonably supply yourself, ask exactly ONE short question "
    "ending with '?' — then act on whatever they say. Never invent specifics (like "
    "an artist or title) the user did not mention. Address the user as \"sir\" "
    "occasionally. /no_think"
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
            # BLANK the un-woken turn before stopping. StopResponse only cancels the
            # reply — the user transcript still lands in the ChatContext, so without
            # this an ambient sentence ("play some music", "open that video") stays in
            # history and gets REPLAYED to the brain on the next turn that DOES wake
            # JARVIS, making it act on things nobody addressed to it. Every downstream
            # consumer (the LLM adapter, memory log, UI transcript) skips empty-text
            # items, so emptying the content makes an un-woken turn a true no-op.
            new_message.content = [""]
            raise StopResponse()
        if rewritten is not None:
            # Strip the wake word so the agent sees just the request.
            new_message.content = [rewritten]
