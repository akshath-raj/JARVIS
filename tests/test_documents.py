"""Tests for document parse / assemble / open-with (offline)."""
from __future__ import annotations

import pytest

from jarvis.documents import openwith
from jarvis.documents.assemble import assemble, detect_format
from jarvis.documents.parse import DocumentError, read_text


def test_detect_format():
    assert detect_format("make it a jupyter notebook") == "ipynb"
    assert detect_format("as a powerpoint") == "pptx"
    assert detect_format("write a python script") == "py"
    assert detect_format("") == "docx"


def test_assemble_ipynb_splits_code_and_prose(tmp_path):
    import nbformat

    ans = "Intro text.\n\n```python\nprint('hi')\n```\n\nOutro."
    p = assemble(ans, "ipynb", str(tmp_path))
    nb = nbformat.read(p, as_version=4)
    kinds = [c.cell_type for c in nb.cells]
    assert "code" in kinds and kinds.count("markdown") == 2


def test_assemble_docx_reparses(tmp_path):
    p = assemble("# Title\n\nHello world.", "docx", str(tmp_path))
    assert "Hello world" in read_text(p)


def test_assemble_py_extracts_code(tmp_path):
    p = assemble("notes\n\n```python\nx = 1\n```", "py", str(tmp_path))
    assert open(p).read().strip() == "x = 1"


def test_open_with_app_maps_and_runs(monkeypatch, tmp_path):
    f = tmp_path / "x.docx"
    f.write_text("x")
    calls = []
    monkeypatch.setattr(openwith, "app_installed", lambda n: True)
    monkeypatch.setattr(
        openwith.subprocess, "run",
        lambda cmd, **k: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    out = openwith.open_with_app(str(f))
    assert calls[0] == ["open", "-a", "Microsoft Word", str(f)]
    assert "Word" in out


def test_open_with_app_falls_back_when_app_missing(monkeypatch, tmp_path):
    f = tmp_path / "x.ipynb"
    f.write_text("{}")
    monkeypatch.setattr(openwith, "app_installed", lambda n: False)  # VS Code "missing"
    calls = []
    monkeypatch.setattr(
        openwith.subprocess, "run",
        lambda cmd, **k: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    openwith.open_with_app(str(f))
    assert calls[0] == ["open", str(f)]  # default open


def test_parse_unsupported_type(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("hi")
    with pytest.raises(DocumentError):
        read_text(f)
