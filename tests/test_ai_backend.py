"""Tests for the AI backends, registry, and the assignment flow (offline)."""
from __future__ import annotations

import asyncio

from jarvis.ai_apps import registry
from jarvis.ai_apps.base import AnswerBackend, BackendError
from jarvis.ai_apps.prompts import build_prompt
from jarvis.documents import flow
from jarvis.documents.workspace import Workspace


def _run(c):
    return asyncio.run(c)


def test_prompt_builder_shapes_by_format():
    p = build_prompt("keep it concise", "ipynb", "hw3.pdf")
    assert "hw3.pdf" in p and "python" in p.lower() and "keep it concise" in p
    assert "Slide" in build_prompt("", "pptx", "x")


def test_registry_first_available_in_order(monkeypatch):
    class Off(AnswerBackend):
        name = "off"

        def available(self):
            return False

        async def generate(self, *a):
            return ""

    class On(AnswerBackend):
        name = "on"

        def available(self):
            return True

        async def generate(self, *a):
            return ""

    monkeypatch.setitem(registry._ALL, "off", Off)
    monkeypatch.setitem(registry._ALL, "on", On)
    assert [b.name for b in registry.available_backends(order=["off", "on"])] == ["on"]


def test_do_assignment_falls_back_then_assembles(monkeypatch, tmp_path):
    ws = Workspace(downloads_dir=str(tmp_path))
    (tmp_path / "a.pdf").write_text("assignment")
    ws.set_assignment(str(tmp_path / "a.pdf"), "an assignment about loops")

    class Fail(AnswerBackend):
        name = "fail"

        def available(self):
            return True

        async def generate(self, *a):
            raise BackendError("nope")

    class Good(AnswerBackend):
        name = "good"

        def available(self):
            return True

        async def generate(self, path, prompt, fmt):
            return "```python\nprint(1)\n```"

    monkeypatch.setattr(flow, "available_backends", lambda: [Fail(), Good()])
    monkeypatch.setattr(flow, "detect_format", lambda *a: "ipynb")
    monkeypatch.setattr(flow, "assemble", lambda ans, fmt, d: str(tmp_path / "answer.ipynb"))
    spoken = []

    async def announce(t):
        spoken.append(t)

    _run(flow.do_assignment("finish as a notebook", ws, announce))
    assert ws.answer_file.endswith("answer.ipynb")
    assert "prepared" in spoken[-1].lower() and "good" in spoken[-1]  # used the fallback backend


def test_do_assignment_without_assignment(tmp_path):
    ws = Workspace(downloads_dir=str(tmp_path))
    spoken = []

    async def announce(t):
        spoken.append(t)

    _run(flow.do_assignment("finish it", ws, announce))
    assert "no assignment" in spoken[0].lower()


def test_do_assignment_all_backends_fail(monkeypatch, tmp_path):
    ws = Workspace(downloads_dir=str(tmp_path))
    (tmp_path / "a.pdf").write_text("x")
    ws.set_assignment(str(tmp_path / "a.pdf"))

    class Fail(AnswerBackend):
        name = "fail"

        def available(self):
            return True

        async def generate(self, *a):
            raise BackendError("down")

    monkeypatch.setattr(flow, "available_backends", lambda: [Fail()])
    monkeypatch.setattr(flow, "detect_format", lambda *a: "docx")
    spoken = []
    _run(flow.do_assignment("do it", ws, lambda t: spoken.append(t) or asyncio.sleep(0)))
    assert "couldn't get the assignment done" in spoken[0].lower()


def test_open_answer(monkeypatch, tmp_path):
    ws = Workspace(downloads_dir=str(tmp_path))
    f = tmp_path / "answer.ipynb"
    f.write_text("{}")
    ws.set_answer(str(f))
    monkeypatch.setattr(flow, "open_with_app", lambda p: "opened answer.ipynb in Visual Studio Code")
    assert "Opening" in flow.open_answer(ws)
    assert "no finished answer" in flow.open_answer(Workspace(downloads_dir=str(tmp_path))).lower()


def test_browser_backend_generate(monkeypatch, tmp_path):
    from jarvis.ai_apps import browser_backend

    async def fake_sync(instruction, files=None, timeout=None):
        assert files == [str(tmp_path / "a.pdf")]
        return {"ok": True, "result": "ANSWER"}

    monkeypatch.setattr(browser_backend, "run_task_sync", fake_sync)
    b = browser_backend.BrowserBackend()
    out = _run(b.generate(str(tmp_path / "a.pdf"), "prompt", "ipynb"))
    assert out == "ANSWER"
