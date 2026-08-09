"""Shared persona + behaviour for every JARVIS agent.

`BaseJarvisAgent` wires the wake-word gate into the STT node so ALL agents (and
every turn after a handoff) respect "Hey Jarvis" without duplicating logic.
"""
from __future__ import annotations

import logging
from typing import AsyncIterable, Optional

from livekit.agents import Agent, ModelSettings, StopResponse, llm, stt

logger = logging.getLogger("jarvis.agent")

# Appended to every agent's instructions. Keeps replies speakable and disables
# qwen3's chain-of-thought for low latency (harmless for cloud models).
VOICE_STYLE = (
    "You are speaking out loud, so keep replies short and natural: a sentence or "
    "two, no markdown, no lists, no emoji. You are JARVIS, a witty, unflappable "
    "British AI butler. Address the user as \"sir\" occasionally, never in every "
    "sentence. /no_think"
)


class BaseJarvisAgent(Agent):
    """Base class that gates incoming audio on the shared wake word."""

    async def stt_node(
        self, audio: AsyncIterable, model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        gate = getattr(self.session.userdata, "wake", None)

        if gate is None or not gate.enabled:
            async for ev in Agent.default.stt_node(self, audio, model_settings):
                yield ev
            return

        async def gated():
            async for frame in audio:
                if gate.accept(frame):
                    yield frame

        async for ev in Agent.default.stt_node(self, gated(), model_settings):
            yield ev

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Relevance gate: stay silent if the utterance isn't directed at JARVIS."""
        gate = getattr(self.session.userdata, "relevance", None)
        if gate is None or not gate.enabled:
            return
        text = (new_message.text_content or "").strip()
        if not await gate.is_directed(turn_ctx, text):
            logger.info("Ignoring undirected utterance: %r", text)
            raise StopResponse()
