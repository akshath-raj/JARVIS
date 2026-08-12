"""Tests for the sandboxed file organiser — confinement, no-delete, dedupe."""
from __future__ import annotations

import pytest

from jarvis.files import FileOrganizer, Sandbox, SandboxError


def _org(tmp_path):
    return FileOrganizer(Sandbox(str(tmp_path)))


# ── sandbox confinement ──────────────────────────────────────────────────────
def test_sandbox_allows_inside_rejects_outside(tmp_path):
    sb = Sandbox(str(tmp_path))
    (tmp_path / "ok.txt").write_text("x")
    assert sb.resolve("ok.txt", must_exist=True).name == "ok.txt"
    for bad in ["/etc/passwd", "../../../../etc/hosts", "/tmp"]:
        with pytest.raises(SandboxError):
            sb.resolve(bad)


def test_move_outside_sandbox_is_blocked(tmp_path):
    org = _org(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(SandboxError):
        org.move("f.txt", "/tmp/escape.txt")


def test_no_delete_api_exists(tmp_path):
    org = _org(tmp_path)
    assert not any(w in name for name in dir(org) for w in ("delete", "remove", "unlink", "rmtree"))


# ── copy / move never overwrite ──────────────────────────────────────────────
def test_move_and_copy_autorename(tmp_path):
    org = _org(tmp_path)
    src = tmp_path / "a.txt"; src.write_text("one")
    dst = tmp_path / "dest"; dst.mkdir()
    (dst / "a.txt").write_text("existing")   # force a collision
    org.copy("a.txt", "dest/")
    names = sorted(p.name for p in dst.iterdir())
    assert names == ["a (1).txt", "a.txt"]   # renamed, original preserved
    assert (dst / "a.txt").read_text() == "existing"  # not overwritten


def test_move_preserves_file_at_destination(tmp_path):
    org = _org(tmp_path)
    (tmp_path / "m.txt").write_text("data")
    (tmp_path / "target").mkdir()
    org.move("m.txt", "target/")
    assert (tmp_path / "target" / "m.txt").read_text() == "data"
    assert not (tmp_path / "m.txt").exists()  # moved, not duplicated


# ── organise: categories + dedupe (no deletion) ──────────────────────────────
def test_organize_sorts_and_sets_aside_duplicates(tmp_path):
    org = _org(tmp_path)
    d = tmp_path / "Downloads"; d.mkdir()
    (d / "pic.png").write_bytes(b"img")
    (d / "notes.txt").write_text("hello")
    (d / "notes_copy.txt").write_text("hello")   # exact duplicate content
    (d / "app.py").write_text("print(1)")

    s = org.organize(str(d))
    assert s["moved"] == 3 and s["duplicates"] == 1
    assert (d / "Images" / "pic.png").exists()
    assert (d / "Code" / "app.py").exists()
    # one of the two identical .txt files lands in Documents, the other is set
    # aside in _Duplicates (which one depends on scan order) — nothing is deleted.
    docs = {p.name for p in (d / "Documents").iterdir()}
    dups = {p.name for p in (d / "_Duplicates").iterdir()}
    assert docs | dups == {"notes.txt", "notes_copy.txt"}
    assert len(docs & {"notes.txt", "notes_copy.txt"}) == 1 and len(dups) == 1


def test_recent_files_ordering(tmp_path):
    import os, time

    org = _org(tmp_path)
    d = tmp_path / "dl"; d.mkdir()
    for i, name in enumerate(["old.txt", "mid.txt", "new.txt"]):
        p = d / name; p.write_text(name)
        os.utime(p, (time.time() + i, time.time() + i))
    recents = org.recent_files(str(d), n=2)
    assert [p.name for p in recents] == ["new.txt", "mid.txt"]
