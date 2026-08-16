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
    def __init__(self):
        self.remembered = []

    def all(self):
        return ["loves drum and bass", "prefers a British voice"]

    def remember(self, text, source="explicit"):
        self.remembered.append(text)
        return "ok"


class _EditableMem:
    """A memory store the HUD can add to and delete from (list-backed)."""
    def __init__(self, items=None):
        self.items = list(items or [])

    def all(self):
        return list(self.items)

    def remember(self, text, source="explicit"):
        self.items.append(text)
        return "ok"

    def remove_exact(self, text):
        if text in self.items:
            self.items.remove(text)
            return True
        return False


def test_add_and_delete_memory_from_hud():
    mem = _EditableMem(["likes tea"])
    ui = UIController(user="A", memory=mem)
    assert ui.add_memory("uses vim") is True
    assert "uses vim" in ui.state(0)["memories"]
    assert ui.add_memory("   ") is False                 # blank rejected
    assert ui.delete_memory("uses vim") is True
    assert "uses vim" not in ui.state(0)["memories"]
    assert ui.delete_memory("not stored") is False       # nothing to delete


def test_memory_routes_add_and_delete(tmp_path):
    from starlette.testclient import TestClient

    from jarvis.ui.server import UIServer

    ui = UIController(user="A", memory=_EditableMem())
    with TestClient(UIServer(ui, port=0)._app()) as c:
        r = c.post("/api/memory/add", json={"text": "I love jazz"})
        assert r.status_code == 200 and r.json()["memories"] == ["I love jazz"]
        r = c.post("/api/memory/add", json={"text": ""})
        assert r.status_code == 400 and r.json()["ok"] is False
        r = c.post("/api/memory/delete", json={"text": "I love jazz"})
        assert r.status_code == 200 and r.json()["memories"] == []
        r = c.post("/api/memory/delete", json={"text": "ghost"})
        assert r.status_code == 404


# ── to-do + agenda deletion from the HUD ────────────────────────────────────
def _task_store(tmp_path):
    from jarvis.scheduler import TaskStore

    return TaskStore(str(tmp_path / "t.json"))


def test_remove_todo_and_agenda_from_hud(tmp_path):
    import time

    s = _task_store(tmp_path)
    t = s.add_todo("water the plants")
    r = s.add_reminder("standup", time.time() + 3600)   # also mirrors a to-do
    e = s.add_event("dentist", time.time() + 7200)
    ui = UIController(user="A", tasks=s)

    assert ui.remove_todo(item_id=t["id"]) is True
    assert "water the plants" not in [x["text"] for x in ui.state(0)["todos"]]
    assert ui.remove_todo(item_id="missing") is False

    # cancelling the reminder clears it from the agenda AND its mirrored to-do
    assert ui.remove_agenda_item(item_id=r["id"], kind="reminder") is True
    agenda = ui.state(0)["agenda"]
    assert all(a.get("id") != r["id"] for a in agenda)
    assert "standup" not in [x["text"] for x in ui.state(0)["todos"]]

    assert ui.remove_agenda_item(item_id=e["id"], kind="event") is True
    assert ui.state(0)["agenda"] == []
    assert ui.remove_agenda_item(item_id="ghost", kind="event") is False


def test_task_delete_routes(tmp_path):
    import time

    from starlette.testclient import TestClient

    from jarvis.ui.server import UIServer

    s = _task_store(tmp_path)
    t = s.add_todo("buy milk")
    e = s.add_event("dentist", time.time() + 3600)
    ui = UIController(user="A", tasks=s)
    with TestClient(UIServer(ui, port=0)._app()) as c:
        r = c.post("/api/todo/delete", json={"id": t["id"]})
        assert r.status_code == 200 and r.json()["todos"] == []
        r = c.post("/api/todo/delete", json={"id": "gone"})
        assert r.status_code == 404 and r.json()["ok"] is False
        r = c.post("/api/agenda/delete", json={"id": e["id"], "kind": "event"})
        assert r.status_code == 200 and r.json()["agenda"] == []
        r = c.post("/api/agenda/delete", json={"id": "gone", "kind": "event"})
        assert r.status_code == 404


# ── first-launch name prompt ────────────────────────────────────────────────
def test_needs_name_when_no_user_set(tmp_path):
    ui = UIController(user="", suggestion="Alex")
    st = ui.state(0)
    assert st["needs_name"] is True        # first launch → HUD should ask
    assert st["name_suggestion"] == "Alex"
    assert st["user"] == ""


def test_known_user_does_not_prompt(tmp_path):
    ui = UIController(user="Akshath")
    assert ui.state(0)["needs_name"] is False


def test_set_user_name_persists_updates_and_teaches_memory(tmp_path, monkeypatch):
    import jarvis.identity as ident

    monkeypatch.setattr(ident, "_PATH", tmp_path / "user.json")
    mem = _Mem()
    ui = UIController(user="", suggestion="Alex", memory=mem)
    saved = ui.set_user_name("  Riya  Sharma ")
    assert saved == "Riya Sharma"                      # trimmed/normalised
    assert ident.load_name() == "Riya Sharma"          # persisted for next session
    assert ui.state(0)["user"] == "Riya Sharma"
    assert ui.state(0)["needs_name"] is False
    assert "The user's name is Riya Sharma." in mem.remembered
    # an "identity" event is emitted so open clients update live
    assert any(e["type"] == "identity" for e in ui.state(0)["events"])


def test_set_user_name_blank_is_ignored(tmp_path, monkeypatch):
    import jarvis.identity as ident

    monkeypatch.setattr(ident, "_PATH", tmp_path / "user.json")
    ui = UIController(user="", suggestion="Alex")
    assert ui.set_user_name("   ") == ""
    assert ui.state(0)["needs_name"] is True            # still needs a name


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
    # a new explanation shows up only after the cursor, as a feed entry with image
    ui.show_explanation(title="Formula", body="x = -b/2a", image_b64="AAAA", source="vision")
    st2 = ui.state(cursor)
    assert [e["type"] for e in st2["events"]] == ["feed"]
    entry = st2["events"][0]["entry"]
    assert entry["kind"] == "explanation" and entry["title"] == "Formula" and entry["image"] == "AAAA"
    assert st2["feed"][-1]["text"] == "x = -b/2a"


def test_qa_turns_build_the_feed_and_dedupe_explanation(tmp_path):
    ui = _ctrl(tmp_path)
    ui.add_turn("user", "what's the capital of France?")
    ui.add_turn("assistant", "Paris, sir.")
    ui.add_turn("bogus", "ignored")     # non user/assistant dropped
    feed = ui.state(0)["feed"]
    assert [(f["kind"], f.get("role"), f["text"]) for f in feed] == [
        ("qa", "user", "what's the capital of France?"),
        ("qa", "assistant", "Paris, sir."),
    ]
    # an answer that merely relays a screen analysis isn't duplicated in the feed
    ui.show_explanation(title="Screen", body="You are looking at VS Code.")
    ui.add_turn("assistant", "You are looking at VS Code.")   # same as the explanation
    kinds = [f["kind"] for f in ui.state(0)["feed"]]
    assert kinds == ["qa", "qa", "explanation"]   # no extra qa entry for the relay


def test_notifier_fires_on_each_event(tmp_path):
    ui = _ctrl(tmp_path)
    seen = []
    ui.set_notifier(lambda ev: seen.append(ev["type"]))
    ui.reveal()
    ui.show_explanation(title="t", body="b")   # emitted as a feed entry now
    ui.navigate("about")
    assert seen == ["reveal", "feed", "navigate"]


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
