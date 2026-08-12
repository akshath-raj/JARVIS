"""Tests for the OpenAI Agents SDK brain + screen vision.

Network (OpenAI) and the OS `screencapture` binary are stubbed, so these run
offline and with no side effects.
"""
from __future__ import annotations

import pytest

from jarvis.tools.screen import ScreenController, ScreenError


# ── ScreenController ────────────────────────────────────────────────────────
def test_capture_returns_path(monkeypatch, tmp_path):
    import subprocess

    def fake_run(cmd, **kwargs):
        # emulate `screencapture` writing a PNG to the target path
        path = cmd[-1]
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sc = ScreenController(vision_model="gpt-4o", openai_api_key="k")
    path = sc.capture()
    assert path.endswith(".png")
    import os
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_capture_raises_on_failure(monkeypatch):
    import subprocess

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "not authorized")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sc = ScreenController(vision_model="gpt-4o", openai_api_key="k")
    with pytest.raises(ScreenError):
        sc.capture()


def test_explain_sends_image_and_returns_answer(monkeypatch, tmp_path):
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    monkeypatch.setattr(ScreenController, "capture", lambda self: str(png))

    captured = {}

    class _Msg:
        content = "You're looking at VS Code with a Python file open, sir."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        chat = _Chat()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    sc = ScreenController(vision_model="gpt-4o", openai_api_key="k")
    ans = sc.explain("what am I looking at?")

    assert "VS Code" in ans
    # the user's question rode along, and an image was attached
    content = captured["messages"][0]["content"]
    assert any(p["type"] == "image_url" for p in content)
    assert "what am I looking at?" in content[0]["text"]
    assert captured["model"] == "gpt-4o"


def test_explain_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # .env may have leaked it into os.environ
    sc = ScreenController(vision_model="gpt-4o", openai_api_key="")
    with pytest.raises(ScreenError):
        sc.explain("hello")


# ── Brain wiring ────────────────────────────────────────────────────────────
class _FakeMemory:
    def prompt_block(self, *a, **k):
        return " [mem]"

    def log_activity(self, *a, **k):
        pass


def _build():
    from jarvis.openai_agent.brain import build_brain

    return build_brain(
        spotify=object(), browser=object(), tavily=object(),
        media=object(), screen=object(), memory=_FakeMemory(), workspace=object(),
    )


def test_brain_has_triage_and_three_specialists():
    triage, _ = _build()
    assert triage.name == "JARVIS"
    names = {a.name for a in triage.handoffs}
    assert names == {"Music", "Browser", "Screen"}


def test_specialists_have_expected_tools():
    triage, _ = _build()
    by_name = {a.name: a for a in triage.handoffs}
    music = {t.name for t in by_name["Music"].tools}
    assert {"play_song", "pause_music", "list_playlists", "recently_played"} <= music
    assert len(music) == 17
    browser = {t.name for t in by_name["Browser"].tools}
    assert {"open_site", "browser_task", "control_video", "download_and_explain"} <= browser
    screen = {t.name for t in by_name["Screen"].tools}
    assert screen == {"explain_screen", "take_screenshot"}
    triage_tools = {t.name for t in triage.tools}
    assert {"web_search", "remember", "forget", "recall_about_me"} == triage_tools


def test_screen_tools_expose_question_arg_and_routing_hint():
    """explain_screen must accept a `question` and advertise screen-reading so the
    triage agent routes 'what's on my screen' to it."""
    from jarvis.openai_agent.tools import build_screen_tools

    tools = {t.name: t for t in build_screen_tools(screen=object(), memory=_FakeMemory())}
    et = tools["explain_screen"]
    assert "screen" in (et.description or "").lower()
    assert "question" in et.params_json_schema.get("properties", {})


# ── Adapter (chat context → agents input) ───────────────────────────────────
def test_adapter_chat_ctx_conversion():
    from livekit.agents.llm import ChatContext

    from jarvis.openai_agent.adapter import _chat_ctx_to_input, _last_user_text

    c = ChatContext.empty()
    c.add_message(role="system", content="ignored")
    c.add_message(role="user", content="play daft punk")
    c.add_message(role="assistant", content="now playing")
    c.add_message(role="user", content="louder")
    items = _chat_ctx_to_input(c)
    assert items == [
        {"role": "user", "content": "play daft punk"},
        {"role": "assistant", "content": "now playing"},
        {"role": "user", "content": "louder"},
    ]
    assert _last_user_text(items) == "louder"
