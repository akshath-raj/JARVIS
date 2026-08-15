"""Wake-word activation with smart follow-up.

Rules:
  * ASLEEP by default — a turn is only answered after a *direct* wake phrase at
    the start of the turn (normally ``hey jarvis``).  A name mentioned in a song,
    video, conversation, or later in a sentence is never a wake event.
  * If JARVIS's reply is a CLARIFICATION QUESTION, a longer follow-up window opens
    and the user's next turn is answered WITHOUT the wake word.
  * If JARVIS's reply is an ANSWER (not a question), a shorter CONTINUATION window
    opens so back-to-back commands work — e.g. "play some music" then, right
    after, "play something calming" — without repeating the wake word. Once the
    window lapses it goes back to sleep and the wake word is required again.

Transcript-based (not acoustic) so both "jarvis" and "hey jarvis" work, and so it
composes with the smart follow-up state.
"""
from __future__ import annotations

import re
import time

_FILLER = {"", "uh", "um", "hmm", "mm", "mhm", "hm", ".", "you", "the"}
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
        followup_seconds: float = 20.0,
        continuation_seconds: float = 0.0,
        strict: bool = False,
    ):
        self.enabled = enabled
        self.require_hey = require_hey
        self.followup_seconds = followup_seconds
        self.continuation_seconds = continuation_seconds
        # Strict mode: EVERY turn must start with the wake word — the follow-up /
        # continuation windows are ignored, so JARVIS never treats ambient speech
        # (you talking to someone else, background chatter) as a command. This is the
        # safe default for an always-on mic; the cost is that answering a rare
        # clarifying question also needs the wake word ("hey jarvis, the blue one").
        self.strict = strict
        self._awake_until = 0.0
        self._last_cmd = ""
        self._last_cmd_at = 0.0
        alt = "|".join(re.escape(w) for w in words)
        # Activate whenever the wake word appears ANYWHERE in the transcript, and treat
        # only what FOLLOWS it as the command — "so, jarvis play some jazz" wakes and
        # runs "play some jazz". A leading "hey" is optional. This is deliberately more
        # permissive than a strict prefix: STT frequently drops or mangles the opening
        # word, so requiring the wake phrase to start the turn made JARVIS miss real
        # commands. The wake word itself is still a hard \b-bounded token, so it won't
        # fire on substrings, and everything before it is discarded.
        self._wake_re = re.compile(rf"(?:\bhey\b[\s,.:;!-]*)?\b(?:{alt})\b[\s,.:;!?-]*", re.I)

    @property
    def awake(self) -> bool:
        return time.time() < self._awake_until

    def reset(self) -> None:
        """Force back to sleep — the wake word is required again. Called when a long
        silence starts a fresh conversation, so a lapsed follow-up window can't be
        ridden by an unrelated later utterance."""
        self._awake_until = 0.0

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

        # In non-strict mode a recent question/answer opens a brief window where the
        # next turn is answered WITHOUT the wake word. In strict mode (default) this
        # window is ignored entirely, so only the wake word ever activates JARVIS.
        if not self.strict and self.awake:
            if text.lower() in _FILLER:
                return False, None
            return True, None

        # asleep -> require the wake word
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

    def note_reply(self, text: str) -> None:
        """Called after JARVIS replies. A question opens the longer follow-up
        window; any other reply opens the shorter continuation window so the user
        can chain commands (e.g. change the music) without repeating the wake word."""
        if (text or "").strip().endswith("?"):
            self._awake_until = time.time() + self.followup_seconds
        elif self.continuation_seconds > 0:
            self._awake_until = time.time() + self.continuation_seconds
        else:
            self._awake_until = 0.0
