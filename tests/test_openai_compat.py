"""Tests for the OpenAI-compatible chat helper (token-param / temperature fallback)."""
from __future__ import annotations

import jarvis.openai_compat as oc
from jarvis.openai_compat import chat_complete


class _Resp:
    pass


class _FakeClient:
    """Records calls; raises the given error the first time a banned param appears."""
    def __init__(self, base_url, *, reject: str = ""):
        self.base_url = base_url
        self.reject = reject          # a param name the "model" doesn't support
        self.calls: list[dict] = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject and self.reject in kwargs:
            raise Exception(
                f"Error code: 400 - Unsupported parameter: '{self.reject}' is not supported"
            )
        return _Resp()


def setup_function():
    oc._TOKEN_PARAM.clear()   # isolate the per-endpoint cache between tests


def test_swaps_to_max_tokens_when_completion_rejected():
    c = _FakeClient("https://api.cerebras.ai/v1", reject="max_completion_tokens")
    chat_complete(c, model="m", messages=[], max_output=100)
    # first attempt uses max_completion_tokens (rejected), retry uses max_tokens
    assert "max_completion_tokens" in c.calls[0]
    assert "max_tokens" in c.calls[1]
    # and the working param is cached → next call goes straight to max_tokens
    chat_complete(c, model="m", messages=[], max_output=50)
    assert "max_tokens" in c.calls[2] and "max_completion_tokens" not in c.calls[2]


def test_drops_temperature_when_unsupported():
    c = _FakeClient("https://api.openai.com/v1", reject="temperature")
    chat_complete(c, model="m", messages=[], max_output=100, temperature=0.2)
    assert any("temperature" not in call for call in c.calls)  # eventually succeeds without it


def test_passes_through_when_all_supported():
    c = _FakeClient("https://api.openai.com/v1")
    chat_complete(c, model="m", messages=[], max_output=100, temperature=0.2)
    assert len(c.calls) == 1
    assert c.calls[0]["max_completion_tokens"] == 100
