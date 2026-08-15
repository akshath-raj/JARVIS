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


def test_brain_is_single_fast_agent():
    """One agent, no handoffs — device actions resolve in a single API call, while
    informational tools keep the loop open so compound requests complete."""
    agent, _ = _build()
    assert agent.name == "JARVIS"
    assert agent.handoffs == []
    # a policy (callable) governs stopping: actions stop, info tools chain
    assert callable(agent.tool_use_behavior)


def test_tool_policy_stops_on_actions_continues_on_info():
    from agents.agent import ToolsToFinalOutputResult

    from jarvis.openai_agent.brain import _tool_final_policy

    class _R:
        def __init__(self, name, output):
            self.tool = type("T", (), {"name": name})()
            self.output = output

    # a pure device action → finalise and speak its result directly
    r = _tool_final_policy(None, [_R("pause_music", "Music paused, sir.")])
    assert isinstance(r, ToolsToFinalOutputResult)
    assert r.is_final_output and r.final_output == "Music paused, sir."

    # an informational tool → keep the loop open so it can chain / synthesise
    r2 = _tool_final_policy(None, [_R("explain_this", "…the graph shows…")])
    assert r2.is_final_output is False

    # mixed turn with any info tool → keep going
    r3 = _tool_final_policy(None, [_R("pause_music", "x"), _R("web_search", "y")])
    assert r3.is_final_output is False


def test_agent_has_all_domain_tools():
    agent, _ = _build()
    names = {t.name for t in agent.tools}
    # music + browser + screen + web/memory all live on the one agent
    assert {"play_song", "pause_music", "change_volume", "recently_played"} <= names
    assert {"open_site", "play_youtube", "play_netflix", "control_video"} <= names
    assert {"explain_screen", "take_screenshot"} <= names
    assert {"web_search", "remember", "forget", "recall_about_me"} <= names


def test_screen_tools_expose_question_arg_and_routing_hint():
    """explain_screen must accept a `question` and advertise screen-reading so the
    triage agent routes 'what's on my screen' to it."""
    from jarvis.openai_agent.tools import build_screen_tools

    tools = {t.name: t for t in build_screen_tools(screen=object(), memory=_FakeMemory())}
    et = tools["explain_screen"]
    assert "screen" in (et.description or "").lower()
    assert "question" in et.params_json_schema.get("properties", {})


def test_explain_this_present_with_rag_and_is_document_aware():
    """With a document index, JARVIS gains explain_this: aware of the live on-screen
    document, its page/slide, and the surrounding subtopic pages."""
    from jarvis.openai_agent.tools import build_screen_tools

    with_rag = {t.name: t for t in
                build_screen_tools(screen=object(), memory=_FakeMemory(), rag=object())}
    assert "explain_this" in with_rag
    d = (with_rag["explain_this"].description or "").lower()
    assert "document" in d and ("page" in d or "subtopic" in d or "slide" in d)
    # no index → no explain_this (nothing to ground it in)
    without = {t.name for t in build_screen_tools(screen=object(), memory=_FakeMemory())}
    assert "explain_this" not in without


def test_summarize_this_present_with_rag_and_summarises_onscreen_doc():
    """With a document index, JARVIS gains summarize_this: it identifies the live
    on-screen document and summarises it without the user naming the file."""
    from jarvis.openai_agent.tools import build_screen_tools

    with_rag = {t.name: t for t in
                build_screen_tools(screen=object(), memory=_FakeMemory(), rag=object())}
    assert "summarize_this" in with_rag
    d = (with_rag["summarize_this"].description or "").lower()
    assert "summar" in d and ("screen" in d or "in front of me" in d)
    # section_only lets the user scope it to just the visible section/slide/page
    assert "section_only" in with_rag["summarize_this"].params_json_schema.get("properties", {})
    # no index → no summarize_this (nothing to summarise from)
    without = {t.name for t in build_screen_tools(screen=object(), memory=_FakeMemory())}
    assert "summarize_this" not in without


def test_browser_task_registered():
    """The multi-step browser task tool is present (no separate streaming tool)."""
    from jarvis.openai_agent.tools import build_browser_tools

    names = {t.name for t in build_browser_tools(browser=object(), memory=_FakeMemory())}
    assert "browser_task" in names
    assert "watch_on_site" not in names


def test_files_tools_open_and_find_by_description():
    """Opening/asking a document by free-text description (no exact filename)."""
    from jarvis.openai_agent.tools import build_files_tools

    names = {t.name for t in
             build_files_tools(rag=object(), organizer=object(), memory=_FakeMemory())}
    assert {"open_document", "open_documents", "find_document",
            "summarize_document", "review_document", "ask_documents"} <= names


def test_review_keeps_loop_open_and_open_plus_analysis_continues():
    """review_document is an info tool (chains for multi-doc feedback); and opening a
    doc keeps the loop open when the same request also asks to review/explain it."""
    from agents.agent import ToolsToFinalOutputResult

    from jarvis.openai_agent.brain import _tool_final_policy

    class _R:
        def __init__(self, name, output="x"):
            self.tool = type("T", (), {"name": name})()
            self.output = output

    class _Ctx:
        def __init__(self, ut):
            self.context = type("C", (), {"user_text": ut})()

    # a review chains (not final) so multiple docs' feedback can be combined
    assert _tool_final_policy(_Ctx(""), [_R("review_document")]).is_final_output is False
    # open + "suggest improvements" in the request → keep going (read then synthesise)
    r = _tool_final_policy(_Ctx("open both my resumes and suggest improvements"),
                           [_R("open_document")])
    assert r.is_final_output is False
    # a plain open with no analysis intent still stops on the action (one round-trip)
    r2 = _tool_final_policy(_Ctx("open my resume"), [_R("open_document", "Opening it, sir.")])
    assert r2.is_final_output and r2.final_output == "Opening it, sir."


def test_all_intent_detection_for_plural_open():
    """"open ALL/every/multiple … docs" is recognised, plain singular is not — this is
    how open_document recovers a plural intent the model dropped from its argument."""
    from jarvis.openai_agent.tools import _wants_all

    assert _wants_all("open all the documents related to natural disaster")
    assert _wants_all("open every disaster management pdf")
    assert _wants_all("pull up both my resumes")
    assert not _wants_all("open my speaker recognition notes")
    assert not _wants_all("open the overall summary")  # 'overall' must not trip 'all'


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


def test_clean_strips_leaked_tool_json_and_harmony_tokens():
    """The spoken text must never contain leaked tool-call JSON or gpt-oss harmony
    control tokens — but real prose (incl. brace literals) stays intact."""
    from jarvis.openai_agent.adapter import _clean

    # tool-call JSON printed as text is removed, surrounding prose kept
    assert _clean('The slide is a PDF.{"tool":"takescreenshot","arguments":{}}It shows notes.') \
        == "The slide is a PDF.It shows notes."
    assert _clean('Here you go, sir. {"name":"web_search","arguments":{"query":"x"}}') \
        == "Here you go, sir."
    # harmony channel/control tokens are stripped
    assert _clean("<|channel|>commentary<|message|>Playing now, sir.") == "Playing now, sir."
    assert _clean("<|start|>assistant<|channel|>final<|message|>Paused, sir.<|end|>") == "Paused, sir."
    assert _clean("Done <tool_call>{\"tool\":\"pause_music\"}</tool_call> paused, sir.") \
        == "Done paused, sir."
    # harmony tool-ROUTING syntax (function names) is stripped, prose kept — both the
    # full channel form and a bare leaked "functions.<name>" token.
    assert _clean("<|channel|>commentary to=functions.play_song<|message|>Now playing, sir.") \
        == "Now playing, sir."
    assert _clean("functions.change_volume Volume up, sir.").strip() == "Volume up, sir."
    # legitimate prose (a real word, a brace literal) is left alone
    assert _clean("I have no commentary on that, sir.") == "I have no commentary on that, sir."
    assert _clean("Kept {a literal} and math {x:1} intact.") == "Kept {a literal} and math {x:1} intact."


def test_adapter_uses_completed_tool_output_when_sdk_omits_final_output():
    """A successful action must still be spoken when a provider leaves
    RunResult.final_output empty."""
    from jarvis.openai_agent.adapter import _spoken_result

    tool_item = type("ToolItem", (), {
        "type": "tool_call_output_item", "output": "Now playing Midnight City, sir."
    })()
    result = type("Result", (), {"final_output": None, "new_items": [tool_item]})()
    assert _spoken_result(result) == "Now playing Midnight City, sir."


def test_adapter_always_has_a_nonempty_voice_fallback():
    from jarvis.openai_agent.adapter import _spoken_result

    result = type("Result", (), {"final_output": None, "new_items": []})()
    assert _spoken_result(result) == "Done, sir."


def test_adapter_speaks_real_answer_over_lazy_filler():
    """If the model shrugs off a real explanation with 'there you have it, sir.', the
    substantial tool answer is spoken instead."""
    from jarvis.openai_agent.adapter import _spoken_result

    answer = ("The Kalman filter tracks an object by predicting its next state from a "
              "motion model, then correcting that prediction with each noisy "
              "measurement, weighting the two by their relative uncertainty. " * 2)
    tool_item = type("ToolItem", (), {"type": "tool_call_output_item", "output": answer})()
    result = type("Result", (), {"final_output": "There you have it, sir.",
                                  "new_items": [tool_item]})()
    assert _spoken_result(result) == answer.strip()


def test_adapter_keeps_a_genuine_short_final_reply():
    """A short, real confirmation is NOT overridden — only content-free filler is."""
    from jarvis.openai_agent.adapter import _spoken_result

    tool_item = type("ToolItem", (), {"type": "tool_call_output_item",
                                       "output": "Opening Kalman Filtering.pdf, sir."})()
    result = type("Result", (), {"final_output": "Opening Kalman Filtering.pdf, sir.",
                                  "new_items": [tool_item]})()
    assert _spoken_result(result) == "Opening Kalman Filtering.pdf, sir."
