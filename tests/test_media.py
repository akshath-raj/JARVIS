"""Tests for live-tab video control — targeting the tab the user is watching."""
from __future__ import annotations

from jarvis.tools.media import MediaController


class _Rec(MediaController):
    """Records AppleScripts; simulates the tab-finder returning `found`."""
    def __init__(self, found: str):
        super().__init__()
        self.scripts: list[str] = []
        self._found = found

    def _osa(self, script: str) -> str:
        self.scripts.append(script)
        if "repeat with w in windows" in script:   # the find-media-tab probe
            return self._found
        return ""

    @property
    def action(self) -> str:
        return [s for s in self.scripts if "repeat with w in windows" not in s][-1]


class _CloseRec(MediaController):
    """Captures the single script and returns a fixed osascript result."""
    def __init__(self, ret: str):
        super().__init__()
        self.script = ""
        self._ret = ret

    def _osa(self, script: str) -> str:
        self.script = script
        return self._ret


def test_close_tabs_matching_builds_script_and_counts():
    m = _CloseRec("3")
    msg = m.close_tabs_matching("VTOP")
    assert 'set q to "vtop"' in m.script                 # query lower-cased + quoted
    assert "URL of t contains q" in m.script
    assert "title of t contains q" in m.script
    assert "by -1" in m.script                            # closes in reverse (index-safe)
    assert msg == "closed 3 VTOP tabs, sir"


def test_close_tabs_matching_reports_when_none_open():
    assert _CloseRec("0").close_tabs_matching("netflix") == "no netflix tabs were open, sir"


def test_close_tabs_matching_requires_a_query():
    import pytest

    from jarvis.tools.media import MediaError
    with pytest.raises(MediaError):
        _CloseRec("0").close_tabs_matching("   ")


def test_control_targets_playing_tab():
    m = _Rec("2,3")
    m.control("pause")
    assert "tab 3 of window 2" in m.action
    assert "v.pause()" in m.action


def test_control_falls_back_to_active_tab_when_no_video():
    m = _Rec("")   # no tab has a video
    m.control("pause")
    assert "active tab of front window" in m.action


def test_find_media_tab_parses_and_guards():
    assert _Rec("1,4")._find_media_tab() == (1, 4)
    assert _Rec("")._find_media_tab() is None
    assert _Rec("garbage")._find_media_tab() is None


def test_aliases_and_setters_route_through_finder():
    m = _Rec("1,1")
    assert "louder" in m.control("louder")            # alias → volume_up
    assert m.control("set_volume", "40") == "video volume set to 40"
    assert "tab 1 of window 1" in m.action
    assert m.control("resume") == "resumed"           # alias → play
