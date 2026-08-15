"""Tests for the Mac system-settings controller (brightness / volume / appearance).

AppleScript and the `brightness` CLI are stubbed, so these run offline without
touching the real machine.
"""
from __future__ import annotations

import jarvis.tools.system_settings as ss
from jarvis.tools.system_settings import SystemSettings


def _patch_osa(monkeypatch, replies=None):
    """Record every AppleScript and return canned replies keyed by substring."""
    scripts: list[str] = []
    replies = replies or {}

    def fake(script: str, timeout: int = 15) -> str:
        scripts.append(script)
        for needle, val in replies.items():
            if needle in script:
                return val
        return ""

    monkeypatch.setattr(ss, "_osa", fake)
    return scripts


def test_set_volume_clamps_and_unmutes(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    s = SystemSettings()
    assert s.set_volume(150) == 100                      # clamped to 100
    assert any("set volume output volume 100" in x for x in scripts)
    assert any("without output muted" in x for x in scripts)  # >0 unmutes


def test_set_volume_zero_does_not_unmute(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    SystemSettings().set_volume(-5)                      # clamps to 0
    assert any("set volume output volume 0" in x for x in scripts)
    assert not any("without output muted" in x for x in scripts)


def test_nudge_volume_reads_then_sets(monkeypatch):
    scripts = _patch_osa(monkeypatch, {"output volume of": "40"})
    assert SystemSettings().nudge_volume(up=True, step=10) == 50
    assert any("set volume output volume 50" in x for x in scripts)


def test_is_dark_mode_parses_true(monkeypatch):
    _patch_osa(monkeypatch, {"get dark mode": "true"})
    assert SystemSettings().is_dark_mode() is True


def test_set_dark_mode_emits_appearance_script(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    SystemSettings().set_dark_mode(True)
    assert any("set dark mode to true" in x for x in scripts)


def test_set_brightness_uses_cli_when_present(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ss.subprocess, "run",
                        lambda a, **k: calls.append(a) or type("R", (), {"stdout": ""})())
    s = SystemSettings()
    s._brightness_cli = "/usr/local/bin/brightness"      # pretend it's installed
    assert s.set_brightness(1.5) == 1.0                   # clamped
    assert calls[-1][0] == "/usr/local/bin/brightness"
    assert calls[-1][1] == "1.00"


def test_set_brightness_falls_back_to_keys(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    s = SystemSettings()
    s._brightness_cli = None                              # no CLI → key presses
    s.set_brightness(0.5)
    joined = "\n".join(scripts)
    assert "key code 145" in joined                       # steps down to a floor first
    assert "key code 144" in joined                       # then up toward the target


def test_night_shift_noop_without_cli(monkeypatch):
    s = SystemSettings()
    s._nightlight_cli = None
    assert s.can_night_shift is False
    assert s.set_night_shift(True) is False               # nothing installed → not applied


def test_night_shift_invokes_cli(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ss.subprocess, "run",
                        lambda a, **k: calls.append(a) or type("R", (), {"stdout": ""})())
    s = SystemSettings()
    s._nightlight_cli = "/opt/homebrew/bin/nightlight"
    assert s.set_night_shift(True) is True
    assert calls[-1] == ["/opt/homebrew/bin/nightlight", "on"]
