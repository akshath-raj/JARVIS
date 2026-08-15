"""Tests for vision+reasoning image vetting. The model calls are stubbed, so these
run offline; we assert the describe→judge→accept pipeline and its fail-open safety.
"""
from __future__ import annotations

import jarvis.openai_compat as oc
from jarvis.tools.image_vetting import ImageVetter
from jarvis.tools.images import ImageResult


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


def _vetter():
    v = ImageVetter(vision_model="gemma", text_model="gpt-oss", api_key="k",
                    base_url="https://api.cerebras.ai/v1")
    v._client = lambda: object()   # never build a real OpenAI client
    return v


def test_describe_returns_vision_text(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nDATA")
    monkeypatch.setattr(oc, "chat_complete",
                        lambda *a, **k: _Resp("A photo of the Hollywood sign on fire."))
    assert "Hollywood" in _vetter().describe(str(img))


def test_judge_parses_fit_json(monkeypatch):
    monkeypatch.setattr(oc, "chat_complete",
                        lambda *a, **k: _Resp('{"fit": false, "reason": "it is a Venn diagram"}'))
    fit, reason = _vetter().judge("a deepfake photo", "Venn.png", "a Venn diagram")
    assert fit is False and "Venn" in reason


def test_accept_rejects_then_the_loop_moves_on(monkeypatch):
    # describe ok; judge says NO → accept() returns False (fetch_one will try next)
    monkeypatch.setattr(oc, "chat_complete", lambda *a, **k: _Resp('{"fit": false}'))
    v = _vetter()
    monkeypatch.setattr(v, "describe", lambda p: "a chart")
    accept = v.accept_for("a deepfake photo")
    res = ImageResult("Chart", "https://upload.wikimedia.org/c.png", "src")
    assert accept("/tmp/c.png", res) is False


def test_accept_fails_open_on_model_error(monkeypatch):
    v = _vetter()
    def boom(p):
        raise RuntimeError("vision down")
    monkeypatch.setattr(v, "describe", boom)
    accept = v.accept_for("anything")
    # a model hiccup must NOT block the document — accept the image
    assert accept("/tmp/x.png", ImageResult("X", "https://upload.wikimedia.org/x.png", "s")) is True


def test_from_config_prefers_cerebras(monkeypatch):
    import types

    from jarvis import config as cfgmod
    fake = types.SimpleNamespace(
        agent_provider="cerebras", cerebras_api_key="ckey",
        cerebras_vision_model="gemma-4-31b", cerebras_model="gpt-oss-120b",
        cerebras_base_url="https://api.cerebras.ai/v1",
        vision_model="gpt-4o", cloud_agent_model="gpt-5.4-mini", openai_api_key="",
    )
    monkeypatch.setattr(cfgmod, "config", fake)
    v = ImageVetter.from_config()
    assert v.vision_model == "gemma-4-31b" and v.text_model == "gpt-oss-120b"
    assert v.enabled
