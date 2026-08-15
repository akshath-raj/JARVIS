"""Tests for the Pages word-processor controller + clipboard.

AppleScript (`_osa`) and the pbcopy/pbpaste subprocess are stubbed, so these run
offline without touching Pages or the real clipboard.
"""
from __future__ import annotations

import pytest

import jarvis.tools.pages as pg
from jarvis.tools.pages import PagesController, PagesError


def _patch_osa(monkeypatch, replies=None):
    """Record every AppleScript and return canned replies keyed by substring."""
    scripts: list[str] = []
    replies = replies or {}

    def fake(script: str, timeout: int = 25) -> str:
        scripts.append(script)
        for needle, val in replies.items():
            if needle in script:
                return val
        return ""

    monkeypatch.setattr(pg, "_osa", fake)
    # neutralise the LaunchServices launch/focus so tests never open the real app
    monkeypatch.setattr(PagesController, "_activate", lambda self: None)
    return scripts


def test_new_document_makes_a_document(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    assert "ready" in PagesController().new_document()
    assert any('make new document' in s for s in scripts)


def test_activate_launches_via_launchservices(monkeypatch):
    """_activate uses `open -a` (reliable) and waits until the app is running —
    NOT AppleScript `activate`, which can fail with -600 before launch."""
    import subprocess as sp
    calls = {}
    monkeypatch.setattr(pg, "_osa", lambda s, timeout=25: "true")  # _is_running → True
    monkeypatch.setattr(sp, "run",
                        lambda cmd, **k: calls.setdefault("cmd", cmd) or
                        sp.CompletedProcess(cmd, 0, "", ""))
    PagesController("Pages")._activate()
    assert calls["cmd"][:2] == ["open", "-a"] and calls["cmd"][2] == "Pages"


def test_get_and_set_text(monkeypatch):
    scripts = _patch_osa(monkeypatch, {"get body text": "hello world"})
    p = PagesController()
    assert p.get_text() == "hello world"
    p.set_text('a "quoted" line\\path')
    # user text is escaped into the AppleScript literal
    assert any('set body text of front document to "a \\"quoted\\" line\\\\path"' in s
               for s in scripts)


def test_append_preserves_existing_text(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    PagesController().append_text("more")
    assert any("(body text of d) &" in s and '"more"' in s for s in scripts)


def test_clear_empties_the_body(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    assert "cleared" in PagesController().clear()
    assert any('set body text of front document to ""' in s for s in scripts)


def test_select_copy_cut_paste_send_the_right_keystrokes(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    p = PagesController()
    p.select_all(); p.copy_selection(); p.cut_selection(); p.paste()
    joined = "\n".join(scripts)
    assert 'keystroke "a" using {command down}' in joined
    assert 'keystroke "c" using {command down}' in joined
    assert 'keystroke "x" using {command down}' in joined
    assert 'keystroke "v" using {command down}' in joined


def test_delete_selection_uses_delete_key(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    PagesController().delete_selection()
    assert any("key code 51" in s for s in scripts)  # Delete


def test_insert_table_clicks_the_insert_menu(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    PagesController().insert_table()
    assert any('menu item "Table" of menu "Insert"' in s for s in scripts)


def test_insert_image_puts_picture_on_clipboard_then_pastes(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    PagesController().insert_image("/tmp/pic.png")
    joined = "\n".join(scripts)
    assert 'read (POSIX file "/tmp/pic.png") as picture' in joined
    assert 'keystroke "v" using {command down}' in joined


def test_add_title_pastes_rtf_and_newline(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    PagesController().add_title("My Report")
    joined = "\n".join(scripts)
    # a title pastes styled RTF from the clipboard, then a Return follows
    assert "as «class RTF »" in joined
    assert "key code 36" in joined


def test_insert_plain_text_uses_clipboard(monkeypatch):
    scripts = _patch_osa(monkeypatch)
    calls = {}
    monkeypatch.setattr(PagesController, "set_clipboard",
                        lambda self, t: calls.setdefault("clip", t))
    PagesController().insert_text("just a paragraph")
    assert calls["clip"] == "just a paragraph"
    assert any('keystroke "v" using {command down}' in s for s in scripts)


def test_permission_error_is_friendly(monkeypatch):
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "System Events got an error: -1743")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(PagesError) as e:
        pg._osa('tell application "Pages" to activate')
    assert "permission" in str(e.value).lower()


def test_clipboard_get_set(monkeypatch):
    import subprocess

    box = {"data": "on the board"}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "pbpaste":
            return subprocess.CompletedProcess(cmd, 0, box["data"], "")
        if cmd[0] == "pbcopy":
            box["data"] = kwargs.get("input", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    p = PagesController()
    assert p.get_clipboard() == "on the board"
    p.set_clipboard("new value")
    assert box["data"] == "new value"
