"""Tests for the two-tier memory store (no Ollama needed).

Explicit facts land in the durable profile (about_user.md); activity/conversation
go to a transient log that a small model distills into the profile every N entries.
A FakeModel stands in for the summarizer so these tests stay fully offline.
"""
from __future__ import annotations

from jarvis.graph.memory import MemoryStore


class _FakeSummarizer:
    """A stand-in chat model whose distilled profile is fixed and deterministic."""

    def __init__(self, bullets):
        self._bullets = bullets

    def invoke(self, _messages):
        class R:
            content = "\n".join(f"- {b}" for b in self._bullets)

        R.content = "\n".join(f"- {b}" for b in self._bullets)
        return R()


def _mem(tmp_path, **kw):
    return MemoryStore(str(tmp_path), **kw)


def test_remember_and_dedup(tmp_path):
    m = _mem(tmp_path)
    assert m.remember("loves jazz").startswith("noted")
    assert m.remember("Loves   Jazz") == "already knew that"  # normalised dup
    assert m.all() == ["loves jazz"]


def test_explicit_goes_to_profile_activity_to_log(tmp_path):
    m = _mem(tmp_path)
    m.remember("name is Akshath")              # explicit → durable profile
    m.log_activity("played music: despacito")  # transient → log, NOT profile
    assert m.all() == ["name is Akshath"]
    assert m.profile_text().strip() == "- name is Akshath"
    assert m._log_count() == 1                 # activity sits in the transient log


def test_forget(tmp_path):
    m = _mem(tmp_path)
    m.remember("is vegetarian")
    m.remember("likes cycling")
    assert m.forget("vegetarian").startswith("forgotten")
    assert "is vegetarian" not in m.all()
    assert m.forget("nonexistent thing").startswith("I have no memory")


def test_profile_persistence_round_trip(tmp_path):
    m = _mem(tmp_path)
    m.remember("name is Akshath")
    m.log_activity("opened youtube")   # transient — should NOT survive as a durable fact
    m2 = MemoryStore(str(tmp_path))    # reload from disk
    assert m2.all() == ["name is Akshath"]


def test_prompt_block_and_recall(tmp_path):
    m = _mem(tmp_path)
    assert m.prompt_block("anything") == ""       # empty when no profile
    assert "don't know anything" in m.recall_text()
    m.remember("loves drum and bass")
    block = m.prompt_block("music")
    assert "loves drum and bass" in block and "user" in block.lower()
    assert "loves drum and bass" in m.recall_text()


def test_log_trims_to_cap(tmp_path):
    import jarvis.graph.memory as mod
    orig = mod._MAX_LOG
    mod._MAX_LOG = 5
    try:
        m = _mem(tmp_path)
        for i in range(20):
            m.log_activity(f"event {i}")
        assert m._log_count() == 5                 # capped, oldest dropped
        details = [e["detail"] for e in m._read_log()]
        assert details == [f"event {i}" for i in range(15, 20)]
    finally:
        mod._MAX_LOG = orig


def test_summarize_updates_profile_and_clears_log(tmp_path):
    model = _FakeSummarizer(["likes sushi", "lives in Bangalore"])
    m = _mem(tmp_path, model=model, summarize_every=3)
    m.log_activity("said: I love sushi")
    m.log_activity("searched: restaurants in Bangalore")
    assert m.summarize_now() is True
    assert set(m.all()) == {"likes sushi", "lives in Bangalore"}  # distilled into profile
    assert m._log_count() == 0                                    # consumed log cleared


def test_summarize_preserves_entries_logged_during_run(tmp_path):
    # _consume_log_locked removes only the N entries the summarizer read; anything
    # appended after that read must remain.
    m = _mem(tmp_path, model=_FakeSummarizer(["likes tea"]), summarize_every=99)
    m.log_activity("said: I like tea")
    entries = m._read_log()
    m.log_activity("said: and coffee")   # a late append after the "read"
    with m._io_lock:
        m._consume_log_locked(len(entries))
    remaining = [e["detail"] for e in m._read_log()]
    assert remaining == ["said: and coffee"]
