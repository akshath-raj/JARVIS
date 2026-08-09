"""Context-aware relevance gate (replaces the wake word).

JARVIS listens to everything, but not every transcript is addressed to it — a lot
is background chatter or noise the STT hallucinated. Before replying, a small, fast
LOCAL model looks at the recent conversation plus the new utterance and decides:
is this actually directed at the assistant?

If not, the agent raises StopResponse and stays silent. It fails OPEN (responds
when unsure) so it never goes unexpectedly mute.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("jarvis.relevance")

_SYSTEM = (
    "You are a filter for a voice assistant named JARVIS. Decide if the LAST user "
    "utterance is a request/question/command DIRECTED AT JARVIS and expecting a "
    "response, versus background speech, side conversation between people, "
    "narration, filler, or noise. Use the conversation so far for context: a "
    "follow-up right after JARVIS asked something IS directed at it.\n"
    "Bias: commands and questions -> yes; declarative statements that are part of a "
    "human-to-human conversation and don't address an assistant -> no.\n"
    "Examples:\n"
    "  'play some jazz' -> yes\n"
    "  'what's the tallest mountain' -> yes\n"
    "  'jarvis, pause' -> yes\n"
    "  'yeah so the quarterly revenue was down again last week' -> no\n"
    "  'haha did you see the game last night' -> no\n"
    "  'um, so anyway' -> no\n"
    "Reply with exactly one word: yes or no."
)

# Obvious noise/filler that never warrants a response.
_FILLER = {"", "uh", "um", "hmm", "mm", "mhm", "hm", "you", "the", "thank you.", "."}


class RelevanceGate:
    def __init__(self, *, enabled: bool, model: str, base_url: str):
        self.enabled = enabled
        self._model = model
        self._client = None
        if not enabled:
            return
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")
            logger.info("Relevance gate active (model: %s)", model)
        except Exception as e:
            logger.warning("Relevance gate unavailable (%s); responding to all.", e)
            self.enabled = False

    async def is_directed(self, turn_ctx, text: str) -> bool:
        """True if the utterance should be answered."""
        norm = text.strip().lower()
        if norm in _FILLER or len(norm) < 2:
            return False
        if self._client is None:
            return True

        transcript = _recent_transcript(turn_ctx)
        prompt = (
            f"Conversation so far:\n{transcript}\n\n"
            f'LAST user utterance: "{text}"\n\n'
            "Is the LAST utterance directed at JARVIS and expecting a response? "
            "Answer yes or no. /no_think"
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4,
            )
            answer = (resp.choices[0].message.content or "").strip().lower()
            answer = answer.replace("<think>", "").replace("</think>", "").strip()
            return not answer.startswith("n")  # fail open: only "no*" suppresses
        except Exception as e:
            logger.warning("Relevance check failed (%s); responding.", e)
            return True


def _recent_transcript(turn_ctx, max_turns: int = 6) -> str:
    items = getattr(turn_ctx, "items", None) or []
    lines: list[str] = []
    for item in items:
        role = getattr(item, "role", None)
        if role not in ("user", "assistant"):
            continue
        txt = (getattr(item, "text_content", "") or "").strip()
        if txt:
            who = "JARVIS" if role == "assistant" else "User"
            lines.append(f"{who}: {txt}")
    return "\n".join(lines[-max_turns:]) or "(nothing yet)"
