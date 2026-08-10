"""Tests for the wake-word + smart-follow-up activation state machine."""
from __future__ import annotations

from jarvis.activation import WakeController


def _c():
    return WakeController(enabled=True, words=("jarvis",), followup_seconds=20)


def test_asleep_requires_wake_word():
    w = _c()
    assert w.gate("play some jazz") == (False, None)


def test_wake_prefix_stripped():
    w = _c()
    assert w.gate("hey jarvis play some jazz") == (True, "play some jazz")
    assert w.gate("jarvis, what is the time") == (True, "what is the time")


def test_name_only_wakes_without_rewrite():
    w = _c()
    assert w.gate("jarvis") == (True, None)


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
    w = WakeController(enabled=True, words=("jarvis",), continuation_seconds=8)
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
