"""Tests for reading mode — orchestration + save/restore, fully offline."""
from __future__ import annotations

from jarvis.tools.reading_mode import ReadingMode
from jarvis.tools.spotify import SpotifyError


class FakeSystem:
    """Records applied settings and returns a fixed 'previous' state on read."""
    def __init__(self, *, night_shift_cli=True):
        self.can_night_shift = night_shift_cli
        self.brightness = 0.9
        self.volume = 70
        self.dark = False
        self.wallpaper = "/old.jpg"
        self.night = False
        self.applied: list[tuple[str, object]] = []

    # reads
    def get_brightness(self): return self.brightness
    def get_volume(self): return self.volume
    def is_dark_mode(self): return self.dark
    def get_wallpaper(self): return self.wallpaper

    # writes (record + mutate)
    def set_brightness(self, f): self.applied.append(("brightness", f)); self.brightness = f
    def set_volume(self, v): self.applied.append(("volume", v)); self.volume = v
    def set_dark_mode(self, on): self.applied.append(("dark", on)); self.dark = on
    def set_night_shift(self, on): self.applied.append(("night", on)); self.night = on; return True
    def set_wallpaper(self, p): self.applied.append(("wallpaper", p)); self.wallpaper = p


class FakeSpotify:
    def __init__(self, fail=False):
        self.played = None
        self.paused = False
        self._fail = fail

    def play_query(self, query, mode, loop=False):
        if self._fail:
            raise SpotifyError("spotify not set up")
        self.played = (query, mode, loop)

    def pause(self):
        self.paused = True


def _mode(**kw):
    sys = kw.pop("system", FakeSystem())
    sp = kw.pop("spotify", FakeSpotify())
    m = ReadingMode(system=sys, spotify=sp, brightness=0.4, volume=20,
                    music_query="calm", **kw)
    return m, sys, sp


def test_start_applies_reading_environment():
    m, sys, sp = _mode()
    msg = m.start()
    assert m.active
    assert ("dark", True) in sys.applied
    assert ("night", True) in sys.applied
    assert ("brightness", 0.4) in sys.applied
    assert ("volume", 20) in sys.applied
    assert sp.played == ("calm", "auto", False)          # default query played
    assert "reading mode on" in msg.lower()


def test_start_twice_is_idempotent():
    m, _, _ = _mode()
    m.start()
    assert "already on" in m.start().lower()


def test_custom_music_query_overrides_default():
    m, _, sp = _mode()
    m.start("lofi beats")
    assert sp.played[0] == "lofi beats"


def test_stop_restores_previous_state():
    m, sys, sp = _mode()
    m.start()
    # simulate the mode having changed live values
    assert sys.dark is True and sys.brightness == 0.4
    msg = m.stop()
    assert not m.active
    assert sys.dark is False                              # restored original
    assert sys.brightness == 0.9
    assert sys.volume == 70
    assert sys.night is False                             # night shift turned back off
    assert sp.paused is True                              # music we started is paused
    assert "restored" in msg.lower()


def test_music_failure_still_enters_mode():
    m, sys, sp = _mode(spotify=FakeSpotify(fail=True))
    msg = m.start()
    assert m.active                                       # settings still applied
    assert ("brightness", 0.4) in sys.applied
    assert sp.played is None                              # music didn't play
    assert "reading mode on" in msg.lower()


def test_stop_without_start():
    m, _, _ = _mode()
    assert "isn't on" in m.stop().lower()


def test_night_shift_skipped_without_cli():
    m, sys, _ = _mode(system=FakeSystem(night_shift_cli=False))
    m.start()
    assert ("night", True) not in sys.applied             # no CLI → not attempted
