"""Tests for the wake-word + smart-follow-up activation state machine."""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.activation import WakeController, current_conversation


@dataclass
class _Item:
    text: str
    created_at: float


def _c():
    return WakeController(enabled=True, words=("jarvis",), require_hey=True, followup_seconds=20)


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


def test_wake_word_anywhere_activates_and_answers_after_it():
    # The wake word need not start the turn: anywhere in the transcript triggers, and
    # only what FOLLOWS it is taken as the command (the lead-in is dropped).
    w = _c()
    assert w.gate("jarvis, play some jazz") == (True, "play some jazz")
    assert w.gate("so um, jarvis what is the time") == (True, "what is the time")
    assert w.gate("okay hey jarvis play some jazz") == (True, "play some jazz")
    # with no wake word at all, nothing is answered
    assert w.gate("play some jazz") == (False, None)
    assert w.gate("what is the time") == (False, None)


def test_wake_word_is_boundary_bounded_not_substring():
    w = _c()
    # a word merely containing the letters (no \b match) must not wake
    assert w.gate("the jarvistic era began") == (False, None)


def test_short_wake_word_can_be_opted_into():
    w = WakeController(enabled=True, words=("jarvis",), require_hey=False)
    assert w.gate("jarvis, play some jazz") == (True, "play some jazz")


def test_question_opens_followup_window():
    w = _c()
    w.note_reply("Which playlist did you mean?")
    assert w.awake is True
    assert w.gate("my focus playlist") == (True, None)  # no wake word needed


def test_answer_returns_to_sleep_by_default():
    # Default (continuation off): a normal answer sends JARVIS back to sleep, so it
    # won't act on ambient audio (e.g. the music it just started) without the wake
    # word. This is the safe default.
    w = _c()
    w.note_reply("Now playing your Focus playlist.")
    assert w.awake is False
    assert w.gate("play more") == (False, None)


def test_continuation_window_is_opt_in():
    # With continuation explicitly enabled (headphones/echo-cancelled setups), a
    # normal answer keeps JARVIS briefly awake so chained commands work.
    w = WakeController(enabled=True, words=("jarvis",), require_hey=True, continuation_seconds=8)
    w.note_reply("Now playing your Focus playlist.")
    assert w.awake is True
    assert w.gate("play something calming") == (True, None)


def test_filler_ignored_even_when_awake():
    w = _c()
    w.note_reply("Anything else?")
    assert w.gate("uh") == (False, None)


def test_disabled_lets_everything_through():
    w = WakeController(enabled=False)
    assert w.gate("random noise") == (True, None)


# ── strict mode (the default): only the wake word ever activates ───────────────
def test_strict_ignores_followup_window():
    # A clarifying question would normally open a no-wake-word window; in strict mode
    # the next turn STILL needs the wake word, so JARVIS can't be triggered by the
    # user talking to someone else right after it asked something.
    w = WakeController(enabled=True, words=("jarvis",), require_hey=True, strict=True)
    w.note_reply("Which playlist did you mean?")
    assert w.gate("my focus playlist") == (False, None)          # ambient, ignored
    assert w.gate("hey jarvis my focus playlist") == (True, "my focus playlist")


def test_strict_ignores_continuation_window():
    w = WakeController(
        enabled=True, words=("jarvis",), require_hey=True,
        continuation_seconds=8, strict=True,
    )
    w.note_reply("Now playing your Focus playlist.")
    assert w.gate("play something calming") == (False, None)     # not woken again


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


def test_reset_sends_back_to_sleep():
    w = _c()
    w.note_reply("Which one?")           # opens follow-up window
    assert w.awake is True
    w.reset()
    assert w.awake is False
    # after reset the wake word is required again
    assert w.gate("play some jazz") == (False, None)


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
