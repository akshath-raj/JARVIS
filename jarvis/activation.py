"""Wake-word activation.

The wake word is required on EVERY turn — there is no follow-up or continuation
window. A turn is answered only if the wake word appears in the transcript; a name
mentioned in a song, video, or a conversation with someone else is never a command,
and JARVIS never stays "awake" after replying. This is deliberate: on an always-on
mic, any window that answers without the wake word lets ambient/other-people speech
(or JARVIS's own reply echoing back) trigger it.

Transcript-based (not acoustic) so both "jarvis" and "hey jarvis" work.
"""
from __future__ import annotations

import re
import time

# Internal transcript emitted by the local speaker gate. It is deliberately not
# a natural-language command, so it can never reach an LLM or any device tool.
VOICE_NOT_RECOGNIZED = "__JARVIS_VOICE_NOT_RECOGNIZED__"


def current_conversation(items: list, *, max_gap: float) -> list:
    """From chat items ordered oldest→newest, return only those belonging to the
    CURRENT conversation — i.e. drop everything before the most recent silence
    longer than ``max_gap`` seconds.

    Each item is expected to expose a ``created_at`` Unix timestamp (LiveKit
    ``ChatMessage`` does). Without this bound, an always-listening session replays
    its entire history every turn, so a command spoken minutes later gets stitched
    onto an earlier "hey jarvis …" turn — mis-priming the turn detector and letting
    the model act on a stale request. Items missing a timestamp are treated as
    contiguous with their neighbour (never a boundary)."""
    items = list(items)
    if max_gap <= 0 or len(items) <= 1:
        return items
    start = 0
    for i in range(1, len(items)):
        prev = getattr(items[i - 1], "created_at", 0) or 0
        cur = getattr(items[i], "created_at", 0) or 0
        if prev and cur and cur - prev > max_gap:
            start = i  # a long silence here → everything before is a past conversation
    return items[start:]


class WakeController:
    def __init__(
        self,
        *,
        enabled: bool = True,
        words: tuple[str, ...] = ("jarvis",),
        require_hey: bool = True,
    ):
        self.enabled = enabled
        self.require_hey = require_hey
        self._last_cmd = ""
        self._last_cmd_at = 0.0
        alt = "|".join(re.escape(w) for w in words)
        # Activate when the wake word appears ANYWHERE in the transcript, and treat
        # only what FOLLOWS it as the command — "so, jarvis play some jazz" wakes and
        # runs "play some jazz". Searching anywhere (not just the start) is deliberate:
        # STT frequently drops/mangles the opening word. The wake word is a hard
        # \b-bounded token, so it won't fire on substrings, and everything before it is
        # discarded.
        #
        # require_hey (default) makes the leading "hey" MANDATORY — the two-word phrase
        # "hey jarvis" is far less likely to appear in ambient speech (you talking to
        # someone else) or in a Whisper mishearing than a bare "jarvis", which is the
        # main source of false wakes. Turn it off (JARVIS_WAKE_REQUIRE_HEY=0) to also
        # accept a lone "jarvis".
        hey = r"\bhey\b[\s,.:;!-]*" if require_hey else r"(?:\bhey\b[\s,.:;!-]*)?"
        self._wake_re = re.compile(rf"{hey}\b(?:{alt})\b[\s,.:;!?-]*", re.I)

    def _detect(self, text: str) -> tuple[bool, str]:
        # Search the whole transcript for the wake word; the command is whatever
        # comes after it (text before it — a lead-in or an unrelated clause — is
        # dropped, so JARVIS only answers the question asked after the wake word).
        m = self._wake_re.search(text)
        if m:
            return True, text[m.end():].strip()
        return False, text

    def gate(self, text: str) -> tuple[bool, str | None]:
        """Decide whether to answer this user turn.

        Returns (allowed, rewritten_text). rewritten_text is the request with the
        wake word stripped (or None to leave the message unchanged).
        """
        text = (text or "").strip()
        if not self.enabled:
            return True, None

        # The wake word is ALWAYS required — there is no awake window. If it isn't in
        # the transcript, this turn is not for JARVIS.
        detected, remainder = self._detect(text)
        if not detected:
            return False, None
        return True, (remainder if remainder else None)

    def is_repeat(self, text: str, *, window: float = 3.0) -> bool:
        """True if `text` is the SAME command just accepted within `window` seconds.

        A single spoken command can surface twice — the STT emitting two finals, an
        echo of JARVIS's own confirmation, or the model firing a tool call twice — and
        acting on it twice opens Netflix twice / bumps the volume twice. Accepted
        commands are remembered briefly so the immediate duplicate is dropped, while a
        deliberate repeat a few seconds later still goes through."""
        norm = " ".join((text or "").lower().split())
        now = time.time()
        if norm and norm == self._last_cmd and now - self._last_cmd_at <= window:
            return True
        self._last_cmd = norm
        self._last_cmd_at = now
        return False
