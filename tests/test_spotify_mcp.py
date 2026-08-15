"""Tests for the optional Spotify MCP server integration (offline — no node spawned).

`build_spotify_mcp_server` only constructs the server object; it never launches the
subprocess (that happens on connect()), so these run without node or Spotify.
"""
from __future__ import annotations

import asyncio
import types

import jarvis.openai_agent.spotify_mcp as sm


def _cfg(**kw):
    base = dict(
        spotify_mcp_enabled=False,
        spotify_mcp_server_path="",
        spotify_mcp_command="node",
        spotify_mcp_cwd="",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(sm, "config", _cfg(spotify_mcp_enabled=False))
    assert sm.build_spotify_mcp_server() is None


def test_enabled_but_missing_path_returns_none(monkeypatch):
    monkeypatch.setattr(sm, "config", _cfg(spotify_mcp_enabled=True, spotify_mcp_server_path=""))
    assert sm.build_spotify_mcp_server() is None


def test_enabled_but_path_not_found_returns_none(monkeypatch):
    monkeypatch.setattr(
        sm, "config", _cfg(spotify_mcp_enabled=True, spotify_mcp_server_path="/no/such/index.js")
    )
    assert sm.build_spotify_mcp_server() is None


def test_enabled_with_valid_path_builds_server(monkeypatch, tmp_path):
    # A real, existing file passes the existence check (contents irrelevant offline).
    build = tmp_path / "build" / "index.js"
    build.parent.mkdir(parents=True)
    build.write_text("// stub")
    monkeypatch.setattr(
        sm, "config",
        _cfg(spotify_mcp_enabled=True, spotify_mcp_server_path=str(build)),
    )
    server = sm.build_spotify_mcp_server()
    assert server is not None
    assert getattr(server, "name", None) == "spotify"


def test_connect_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(sm, "build_spotify_mcp_server", lambda: None)
    assert asyncio.run(sm.connect_spotify_mcp()) is None


def test_connect_swallows_failure(monkeypatch):
    class BadServer:
        async def connect(self):
            raise RuntimeError("node not found")

    monkeypatch.setattr(sm, "build_spotify_mcp_server", lambda: BadServer())
    # A failed connect must not raise — JARVIS carries on without the server.
    assert asyncio.run(sm.connect_spotify_mcp()) is None


def test_connect_returns_connected_server(monkeypatch):
    class OkServer:
        connected = False
        async def connect(self):
            self.connected = True

    srv = OkServer()
    monkeypatch.setattr(sm, "build_spotify_mcp_server", lambda: srv)
    out = asyncio.run(sm.connect_spotify_mcp())
    assert out is srv and srv.connected is True
