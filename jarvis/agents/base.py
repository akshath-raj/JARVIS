"""Shared persona + activation behaviour for every JARVIS agent.

`BaseJarvisAgent` enforces wake-word activation with smart follow-up (see
jarvis.activation) uniformly across all agents, including after a handoff.
"""
from __future__ import annotations

import logging

from livekit.agents import Agent, StopResponse, llm

from jarvis.activation import VOICE_NOT_RECOGNIZED, current_conversation
from jarvis.config import config

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
        if new_message.text_content == VOICE_NOT_RECOGNIZED:
            # This is an authenticated local control signal from SpeakerVerifiedSTT,
            # not user text. Reply deterministically and never let it reach the LLM.
            await self.session.say("Voice not recognized.")
            raise StopResponse()
        ctl = getattr(self.session.userdata, "activation", None)
        if ctl is None:
            return
        # A long silence starts a NEW conversation. Drop the stale turns from the
        # running chat context so this command can't be stitched onto an earlier
        # "hey jarvis …" turn (which mis-primes the turn detector and can make JARVIS
        # act on a minutes-old request — the "random music" bug), and require the
        # wake word again rather than riding a lapsed follow-up window.
        gap = config.conversation_gap_seconds
        if gap > 0:
            try:
                kept = current_conversation(turn_ctx.items, max_gap=gap)
                if len(kept) < len(turn_ctx.items):
                    turn_ctx.items[:] = kept
                    ctl.reset()
            except Exception:  # never let context pruning break the turn
                pass
        allowed, rewritten = ctl.gate(new_message.text_content or "")
        if not allowed:
            logger.info("Ignoring (not woken): %r", new_message.text_content)
            raise StopResponse()
        command = rewritten if rewritten is not None else (new_message.text_content or "")
        # Drop an immediate duplicate of the command we just acted on. One spoken
        # request can surface twice (double STT final, an echo of JARVIS's own reply,
        # a repeated tool call) — acting twice opens Netflix twice or bumps the volume
        # twice. A deliberate repeat a few seconds later still gets through.
        if ctl.is_repeat(command):
            logger.info("Ignoring (duplicate command): %r", command)
            raise StopResponse()
        if rewritten is not None:
            # Strip the wake word so the agent sees just the request.
            new_message.content = [rewritten]
        # Learn ONLY from commands the user actually addressed to JARVIS (woken),
        # not from every ambient utterance the mic picks up. The command text with
        # the wake word already stripped is what carries the preference signal;
        # bare control commands are filtered out inside `log_command`.
        memory = getattr(self.session.userdata, "memory", None)
        if memory is not None:
            try:
                memory.log_command(command)
            except Exception:  # learning must never break the turn
                pass
