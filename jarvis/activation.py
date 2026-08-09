"""Wake-word activation with smart follow-up.

Rules:
  * ASLEEP by default — a turn is only answered if it contains the wake word
    ("jarvis" or "hey jarvis"); the wake word is stripped before the agent sees
    the request.
  * If JARVIS's reply is a CLARIFICATION QUESTION, a short follow-up window opens
    and the user's next turn is answered WITHOUT the wake word.
  * If JARVIS's reply is an ANSWER (not a question), it goes back to sleep — the
    user must say the wake word again.

Transcript-based (not acoustic) so both "jarvis" and "hey jarvis" work, and so it
composes with the smart follow-up state.
"""
from __future__ import annotations

import re
import time

_FILLER = {"", "uh", "um", "hmm", "mm", "mhm", "hm", ".", "you", "the"}


class WakeController:
    def __init__(
        self,
        *,
        enabled: bool = True,
        words: tuple[str, ...] = ("jarvis",),
        followup_seconds: float = 20.0,
    ):
        self.enabled = enabled
        self.followup_seconds = followup_seconds
        self._awake_until = 0.0
        alt = "|".join(re.escape(w) for w in words)
        # Wake word as a leading prefix ("hey jarvis, play ...") ...
        self._prefix = re.compile(rf"^\s*(?:hey\s+|ok(?:ay)?\s+)?(?:{alt})\b[\s,.:;!-]*", re.I)
        # ... or anywhere in the utterance.
        self._any = re.compile(rf"\b(?:hey\s+)?(?:{alt})\b", re.I)

    @property
    def awake(self) -> bool:
        return time.time() < self._awake_until

    def _detect(self, text: str) -> tuple[bool, str]:
        m = self._prefix.match(text)
        if m:
            return True, text[m.end():].strip()
        if self._any.search(text):
            return True, self._any.sub("", text, count=1).strip()
        return False, text

    def gate(self, text: str) -> tuple[bool, str | None]:
        """Decide whether to answer this user turn.

        Returns (allowed, rewritten_text). rewritten_text is the request with the
        wake word stripped (or None to leave the message unchanged).
        """
        text = (text or "").strip()
        if not self.enabled:
            return True, None

        if self.awake:  # follow-up window open (JARVIS just asked a question)
            if text.lower() in _FILLER:
                return False, None
            return True, None

        # asleep -> require the wake word
        detected, remainder = self._detect(text)
        if not detected:
            return False, None
        return True, (remainder if remainder else None)

    def note_reply(self, text: str) -> None:
        """Called after JARVIS replies: stay awake only if it asked a question."""
        if (text or "").strip().endswith("?"):
            self._awake_until = time.time() + self.followup_seconds
        else:
            self._awake_until = 0.0
