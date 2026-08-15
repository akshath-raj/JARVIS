"""Tests for the wake-word activation gate (wake word required on every turn)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from livekit.agents import StopResponse

from jarvis.activation import WakeController, current_conversation
from jarvis.agents.base import BaseJarvisAgent


@dataclass
class _Item:
    text: str
    created_at: float


def _c():
    return WakeController(enabled=True, words=("jarvis",), require_hey=True)


def test_asleep_requires_wake_word():
    w = _c()
    assert w.gate("play some jazz") == (False, None)


def test_wake_prefix_stripped():
    w = _c()
    assert w.gate("hey jarvis play some jazz") == (True, "play some jazz")
    assert w.gate("Hey JARVIS, what is the time") == (True, "what is the time")
    assert w.gate("Hey, Jarvis. what is the time") == (True, "what is the time")


def test_direct_name_only_wakes_without_rewrite():
    w = _c()
    assert w.gate("hey jarvis") == (True, None)


def test_wake_phrase_anywhere_activates_and_answers_after_it():
    # The wake phrase need not start the turn: "hey jarvis" anywhere in the transcript
    # triggers, and only what FOLLOWS it is taken as the command (the lead-in dropped).
    w = _c()
    assert w.gate("okay hey jarvis play some jazz") == (True, "play some jazz")
    assert w.gate("so um, hey jarvis what is the time") == (True, "what is the time")
    # with no wake phrase at all, nothing is answered
    assert w.gate("play some jazz") == (False, None)
    assert w.gate("what is the time") == (False, None)


def test_require_hey_ignores_bare_wake_word():
    # With require_hey (default), a bare "jarvis" — the main source of false wakes from
    # ambient speech or Whisper mishearings — must NOT activate; only "hey jarvis" does.
    w = _c()
    assert w.gate("jarvis, play some jazz") == (False, None)
    assert w.gate("so um, jarvis what is the time") == (False, None)
    assert w.gate("yeah I told jarvis about it earlier") == (False, None)
    assert w.gate("hey jarvis play some jazz") == (True, "play some jazz")


def test_wake_word_is_boundary_bounded_not_substring():
    w = _c()
    # a word merely containing the letters (no \b match) must not wake
    assert w.gate("the jarvistic era began") == (False, None)


def test_short_wake_word_can_be_opted_into():
    w = WakeController(enabled=True, words=("jarvis",), require_hey=False)
    assert w.gate("jarvis, play some jazz") == (True, "play some jazz")


def test_wake_word_required_every_turn_no_followup():
    # There is NO follow-up window: even right after JARVIS asks a clarifying
    # question, the user's next turn STILL needs the wake word. A reply to the
    # question without "jarvis" is treated as ambient and ignored.
    w = _c()
    assert w.gate("my focus playlist") == (False, None)          # no wake word → ignored
    assert w.gate("hey jarvis my focus playlist") == (True, "my focus playlist")


def test_no_awake_state_after_any_reply():
    # The controller keeps no "awake" state at all — a bare command never goes
    # through, no matter what JARVIS said previously.
    w = _c()
    assert w.gate("play more") == (False, None)
    assert w.gate("play something calming") == (False, None)


def test_disabled_lets_everything_through():
    w = WakeController(enabled=False)
    assert w.gate("random noise") == (True, None)


def test_is_repeat_drops_immediate_duplicate():
    w = _c()
    assert w.is_repeat("open netflix") is False   # first time → act
    assert w.is_repeat("open netflix") is True    # instant duplicate → skip
    assert w.is_repeat("open youtube") is False   # a different command → act


def test_is_repeat_allows_deliberate_repeat_after_window():
    w = _c()
    assert w.is_repeat("next song") is False
    w._last_cmd_at -= 10  # simulate the dedup window having elapsed
    # a deliberate repeat after the window is NOT a duplicate — act on it again
    assert w.is_repeat("next song") is False


# ── conversation recency bounding (the "random music" fix) ────────────────────
def test_current_conversation_drops_turns_before_a_long_silence():
    # the exact failure mode: an old woken exchange, then a 2-minute gap, then an
    # unrelated ambient sentence. Only the sentence after the gap should survive.
    items = [
        _Item("hey jarvis pause the music", 1000.0),
        _Item("paused", 1001.0),
        _Item("resume spotify", 1005.0),
        _Item("resumed", 1006.0),
        _Item("yes i rock rockfall is also landsley", 1006.0 + 120),  # 2 min later
    ]
    kept = current_conversation(items, max_gap=45)
    assert [i.text for i in kept] == ["yes i rock rockfall is also landsley"]


def test_current_conversation_keeps_a_live_back_and_forth():
    # turns within the gap window are all one conversation — nothing dropped.
    items = [
        _Item("hey jarvis play jazz", 2000.0),
        _Item("playing jazz", 2001.0),
        _Item("play something calming", 2004.0),
    ]
    assert current_conversation(items, max_gap=45) == items


def test_current_conversation_resets_at_each_new_gap():
    # only the MOST RECENT silence matters: everything before the last big gap goes.
    items = [
        _Item("old one", 0.0),
        _Item("old reply", 1.0),
        _Item("newer one", 100.0),     # gap #1 (before this)
        _Item("newer reply", 101.0),
        _Item("latest", 300.0),        # gap #2 (before this) → keep from here
    ]
    kept = current_conversation(items, max_gap=45)
    assert [i.text for i in kept] == ["latest"]


def test_current_conversation_noop_when_disabled_or_short():
    items = [_Item("a", 0.0), _Item("b", 1000.0)]
    assert current_conversation(items, max_gap=0) == items      # bound disabled
    assert current_conversation(items[:1], max_gap=45) == items[:1]  # single item


# ── the wake GATE at the agent layer (blanking un-woken turns) ─────────────────
class _Msg:
    """Minimal stand-in for a LiveKit ChatMessage."""
    def __init__(self, text):
        self.content = [text]

    @property
    def text_content(self):
        return "".join(c for c in self.content if isinstance(c, str))


def _fake_agent(ctl):
    return SimpleNamespace(session=SimpleNamespace(userdata=SimpleNamespace(activation=ctl)))


def test_ungated_turn_is_blanked_so_it_cannot_leak():
    # An ambient turn (no wake word) must be STOPPED *and* emptied — otherwise it
    # lingers in the ChatContext and gets replayed to the brain on the next woken
    # turn, making JARVIS act on things nobody addressed to it.
    ctl = WakeController(enabled=True, words=("jarvis",), require_hey=False)
    msg = _Msg("play some music")
    with pytest.raises(StopResponse):
        asyncio.run(BaseJarvisAgent.on_user_turn_completed(_fake_agent(ctl), None, msg))
    assert msg.text_content == ""   # blanked → every downstream consumer skips it


def test_woken_turn_is_rewritten_without_wake_word():
    ctl = WakeController(enabled=True, words=("jarvis",), require_hey=False)
    msg = _Msg("jarvis play some jazz")
    asyncio.run(BaseJarvisAgent.on_user_turn_completed(_fake_agent(ctl), None, msg))
    assert msg.text_content == "play some jazz"   # woken, wake word stripped
