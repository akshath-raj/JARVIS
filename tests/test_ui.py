"""Tests for the hidden HUD dashboard — conversation log, controller events, and
the dashboard tools. No network or browser; the server itself isn't booted here."""
from __future__ import annotations

from jarvis.ui import ConversationLog, UIController


# ── ConversationLog ─────────────────────────────────────────────────────────
def test_conversation_log_groups_into_sessions(tmp_path):
    cl = ConversationLog(str(tmp_path / "conv.jsonl"))
    cl.add("user", "play some music")
    cl.add("assistant", "Now playing, sir.")
    cl.add("bogus", "ignored")   # non user/assistant dropped
    cl.add("user", "")           # empty dropped
    s = cl.sessions()
    assert len(s) == 1
    assert s[0]["title"] == "play some music"
    assert [t["role"] for t in s[0]["turns"]] == ["user", "assistant"]


# ── UIController ────────────────────────────────────────────────────────────
class _Mem:
    def all(self):
        return ["loves drum and bass", "prefers a British voice"]


def _ctrl(tmp_path):
    cl = ConversationLog(str(tmp_path / "c.jsonl"))
    cl.add("user", "hi")
    return UIController(user="Akshath", memory=_Mem(), conversations=cl)


def test_state_exposes_profile_and_conversations(tmp_path):
    ui = _ctrl(tmp_path)
    st = ui.state(0)
    assert st["user"] == "Akshath"
    assert st["revealed"] is False
    assert st["memories"] == ["loves drum and bass", "prefers a British voice"]
    assert len(st["conversations"]) == 1


def test_reveal_and_events_are_incremental(tmp_path):
    ui = _ctrl(tmp_path)
    ui.reveal()
    st1 = ui.state(0)
    assert st1["revealed"] is True
    assert [e["type"] for e in st1["events"]] == ["reveal"]
    cursor = st1["cursor"]
    # nothing new since cursor
    assert ui.state(cursor)["events"] == []
    # a new explanation shows up only after the cursor
    ui.show_explanation(title="Formula", body="x = -b/2a", image_b64="AAAA", source="vision")
    st2 = ui.state(cursor)
    assert [e["type"] for e in st2["events"]] == ["explanation"]
    e = st2["events"][0]
    assert e["title"] == "Formula" and e["image"] == "AAAA"
    assert st2["last_explanation"]["body"] == "x = -b/2a"


def test_navigate_reveals_and_emits(tmp_path):
    ui = _ctrl(tmp_path)
    ui.navigate("conversations", item_id=123)
    st = ui.state(0)
    assert st["revealed"] is True
    ev = st["events"][-1]
    assert ev["type"] == "navigate" and ev["section"] == "conversations" and ev["item"] == 123


# ── dashboard tools ─────────────────────────────────────────────────────────
def test_build_ui_tools_present():
    from jarvis.openai_agent.tools import build_ui_tools

    ui = UIController(user="A", memory=_Mem(), conversations=None)
    names = {t.name for t in build_ui_tools(ui=ui, open_cb=lambda: None)}
    assert names == {"show_dashboard", "hide_dashboard", "open_dashboard_section", "display_on_dashboard"}


def test_screen_analyse_returns_answer_and_image(monkeypatch, tmp_path):
    from jarvis.tools.screen import ScreenController

    png = tmp_path / "s.png"

    def _capture(self):  # analyse() deletes the temp shot, so recreate it each call
        png.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
        return str(png)

    monkeypatch.setattr(ScreenController, "capture", _capture)

    class _Msg: content = "A quadratic formula slide."
    class _Choice: message = _Msg()
    class _Resp: choices = [_Choice()]
    class _Comp:
        def create(self, **k): return _Resp()
    class _Chat: completions = _Comp()
    class _Client:
        def __init__(self, **k): pass
        chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    sc = ScreenController(vision_model="gpt-4o", openai_api_key="k")
    answer, image = sc.analyse("explain the formula")
    assert "quadratic" in answer.lower()
    assert image  # base64 of the screenshot came back for the HUD
    # explain() still returns just the string
    assert isinstance(sc.explain("x"), str)
