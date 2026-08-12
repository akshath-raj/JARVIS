"""Tests for the local Whisper STT hallucination filter."""
from __future__ import annotations

from jarvis.plugins.whisper_stt import _dehallucinate


def test_drops_repetition_loop():
    # the exact failure mode: Whisper looping on ambiguous audio
    assert _dehallucinate("Ino" + "doro" * 120) == ""
    assert _dehallucinate("la " * 80) == ""


def test_keeps_normal_speech():
    for s in [
        "pause the music",
        "Jarvis, play some drum and bass please.",
        "what's the weather in London right now and will it rain tomorrow",
        "remind me to submit my assignment at nine pm today",
    ]:
        assert _dehallucinate(s) == s


def test_short_text_always_kept():
    assert _dehallucinate("hi hi hi") == "hi hi hi"   # too short to judge
