"""Tests for the JARVIS MCP server — the tool-belt exposed to any MCP client.

Only checks assembly (which tools register, platform gating); the underlying
controllers are covered by their own tests.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from jarvis.mcp import server as mcp_server


def _tool_names(server) -> set[str]:
    return {t.name for t in asyncio.run(server.list_tools())}


def test_build_server_registers_core_portable_tools(monkeypatch):
    # memory + document + files groups need no API keys, so they always register.
    monkeypatch.setattr(mcp_server, "_local_tools_enabled", lambda: False)
    names = _tool_names(mcp_server.build_server())
    assert {"remember", "forget", "recall_about_me"} <= names
    assert {"ask_documents", "summarize_document", "review_document"} <= names
    assert {"list_folder", "move_file", "copy_file"} <= names


def test_cloud_mode_hides_macos_host_tools(monkeypatch):
    # with local tools OFF (a cloud host), Chrome/AppleScript tools must NOT appear.
    monkeypatch.setattr(mcp_server, "_local_tools_enabled", lambda: False)
    names = _tool_names(mcp_server.build_server())
    assert not ({"open_site", "play_youtube", "play_netflix", "control_video",
                 "close_tabs"} & names)


def test_local_mode_exposes_macos_host_tools(monkeypatch):
    monkeypatch.setattr(mcp_server, "_local_tools_enabled", lambda: True)
    names = _tool_names(mcp_server.build_server())
    assert {"open_site", "control_video", "close_tabs"} <= names


def test_local_tools_gate_defaults_to_platform(monkeypatch):
    # no override → follows the host OS; explicit override wins.
    monkeypatch.delenv("JARVIS_MCP_LOCAL_TOOLS", raising=False)
    monkeypatch.setattr(mcp_server, "_IS_MAC", True)
    assert mcp_server._local_tools_enabled() is True
    monkeypatch.setattr(mcp_server, "_IS_MAC", False)
    assert mcp_server._local_tools_enabled() is False
    monkeypatch.setenv("JARVIS_MCP_LOCAL_TOOLS", "1")
    assert mcp_server._local_tools_enabled() is True
