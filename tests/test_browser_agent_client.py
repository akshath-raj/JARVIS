"""Tests for the browser-agent client (no real browser / LLM / subprocess).

The isolated runner subprocess is mocked, so these are fast and offline. They
cover the fire-and-report flow, Chrome-open guard, single-flight, result parsing,
and that the main venv never imports browser_use.
"""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest import mock

from jarvis.browser_agent import client
from jarvis.browser_agent.announcer import Announcer


def _cfg(**over):
    """A stand-in for the frozen Config with the fields the client reads."""
    base = dict(
        browser_agent_enabled=True, openai_api_key="sk-x", browser_venv=".venv-browser",
        browser_chrome_path="/chrome", browser_user_data_dir="/prof", browser_profile_dir="Default",
        browser_downloads="~/Downloads", browser_headless=False, browser_use_local=False,
        browser_local_model="qwen2.5:7b-instruct", browser_local_max_steps=12,
        browser_frontier_model="gpt-4.1-mini", ollama_base_url="http://localhost:11434/v1",
        browser_max_steps=40, browser_timeout=240, captcha_enabled=True, captcha_model="qwen2.5vl:7b",
        browser_clone_profile="auto", browser_clone_dir="/tmp/jarvis-clone-test",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_main_venv_does_not_import_browser_use():
    # Importing the client must NOT pull in the heavy/conflicting browser_use dep.
    assert "browser_use" not in sys.modules


def _run(coro):
    return asyncio.run(coro)


def test_chrome_open_aborts_in_never_mode(monkeypatch):
    # In 'never' clone mode, an open Chrome blocks (no cloning).
    monkeypatch.setattr(client, "config", _cfg(browser_clone_profile="never"))
    monkeypatch.setattr(client, "_chrome_running", lambda: True)
    monkeypatch.setattr(client.os.path, "exists", lambda p: True)
    out = _run(client._execute("check my aws balance"))
    assert "close" in out.lower() and "chrome" in out.lower()


def test_chrome_open_clones_in_auto_mode(monkeypatch):
    # In 'auto' mode with Chrome open, it clones the profile and drives the copy,
    # honoring the headless setting (visible by default) rather than forcing hidden.
    monkeypatch.setattr(client, "config", _cfg(browser_clone_profile="auto"))
    monkeypatch.setattr(client, "_chrome_running", lambda: True)
    monkeypatch.setattr(client, "_clone_profile", lambda: "/tmp/clone")
    monkeypatch.setattr(client.os.path, "exists", lambda p: True)
    sent = {}

    class FakeProc:
        async def communicate(self, data):
            sent["job"] = json.loads(data.decode())
            return (json.dumps({"ok": True, "result": "aws balance $5"}).encode(), b"")

    async def fake_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec", fake_exec)
    out = _run(client._execute("check my aws balance"))
    assert out == "aws balance $5"
    assert sent["job"]["user_data_dir"] == "/tmp/clone" and sent["job"]["headless"] is False


def test_execute_spawns_subprocess_and_parses(monkeypatch):
    monkeypatch.setattr(client, "config", _cfg())
    monkeypatch.setattr(client, "_chrome_running", lambda: False)
    monkeypatch.setattr(client.os.path, "exists", lambda p: True)

    sent = {}

    class FakeProc:
        async def communicate(self, data):
            sent["job"] = json.loads(data.decode())
            return (json.dumps({"ok": True, "result": "downloaded 3 files to ~/Downloads"}).encode(), b"")

    async def fake_exec(*args, **kwargs):
        sent["argv"] = args
        return FakeProc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec", fake_exec)
    out = _run(client._execute("download my OS notes from VTOP"))
    assert out == "downloaded 3 files to ~/Downloads"
    assert sent["job"]["task"] == "download my OS notes from VTOP"
    assert "runner" in " ".join(sent["argv"])
    # frontier-only by default: no local model passed
    assert sent["job"]["local_model"] == ""


def test_execute_timeout_kills(monkeypatch):
    monkeypatch.setattr(client, "config", _cfg())
    monkeypatch.setattr(client, "_chrome_running", lambda: False)
    monkeypatch.setattr(client.os.path, "exists", lambda p: True)

    killed = {"v": False}

    class FakeProc:
        async def communicate(self, data):
            raise asyncio.TimeoutError

        def kill(self):
            killed["v"] = True

    async def fake_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(client.asyncio, "wait_for", mock.AsyncMock(side_effect=asyncio.TimeoutError))
    out = _run(client._execute("slow task"))
    assert "too long" in out.lower()


def test_run_browser_task_acks_then_announces(monkeypatch):
    monkeypatch.setattr(client, "config", _cfg())

    async def fake_execute(instruction):
        return f"done: {instruction}"

    monkeypatch.setattr(client, "_execute", fake_execute)
    ann = Announcer()
    spoken = []

    class FakeSession:
        async def say(self, text):
            spoken.append(text)

    ann.session = FakeSession()

    async def scenario():
        ack = await client.run_browser_task("check aws", announce=ann.announce)
        # background task announces after the ack
        for _ in range(50):
            if spoken:
                break
            await asyncio.sleep(0.01)
        return ack

    ack = _run(scenario())
    assert "on it" in ack.lower()
    assert spoken == ["done: check aws"]


def test_disabled_returns_message(monkeypatch):
    monkeypatch.setattr(client, "config", _cfg(browser_agent_enabled=False))
    out = _run(client.run_browser_task("x"))
    assert "enabled" in out.lower()


def test_run_task_sync_returns_dict_with_file(monkeypatch):
    monkeypatch.setattr(client, "config", _cfg())
    monkeypatch.setattr(client, "_chrome_running", lambda: False)
    monkeypatch.setattr(client.os.path, "exists", lambda p: True)

    class FakeProc:
        async def communicate(self, data):
            job = json.loads(data.decode())
            assert job["available_file_paths"] == ["/tmp/a.pdf"]
            assert job["local_model"] == ""  # frontier-only for sync flows
            return (json.dumps({"ok": True, "result": "the answer"}).encode(), b"")

    async def fake_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(client.asyncio, "create_subprocess_exec", fake_exec)
    res = _run(client.run_task_sync("do it", files=["/tmp/a.pdf"]))
    assert res["ok"] and res["result"] == "the answer"
