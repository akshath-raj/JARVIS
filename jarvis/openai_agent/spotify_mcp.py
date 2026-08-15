"""Optional integration of the external Spotify MCP server
(https://github.com/marcelmarais/spotify-mcp-server).

JARVIS keeps its own AppleScript-based Spotify controller as the DEFAULT — it runs
the desktop app in the background, never steals focus, and covers play/pause/skip/
volume without needing the Web API. This module OPTIONALLY attaches the Node MCP
server to the OpenAI-Agents brain to add the richer operations JARVIS lacks:
queue management, playlist reordering, saving albums, and Spotify Connect device
control. It is gated by JARVIS_SPOTIFY_MCP=1 plus a path to the built server.

One-time setup (see README):
    git clone https://github.com/marcelmarais/spotify-mcp-server
    cd spotify-mcp-server && npm install && npm run build && npm run auth
    # then set JARVIS_SPOTIFY_MCP=1 and JARVIS_SPOTIFY_MCP_PATH=<repo>/build/index.js

The server drives the Spotify Web API, so its playback/volume tools require Spotify
Premium and an active Spotify device. Everything here degrades gracefully: if the
server is disabled, unconfigured, or fails to start, JARVIS runs exactly as before.
"""
from __future__ import annotations

import logging
import os

from jarvis.config import config

logger = logging.getLogger("jarvis.openai_agent.spotify_mcp")


def build_spotify_mcp_server():
    """Return an UNCONNECTED MCP server for the Spotify MCP, or None if disabled /
    unconfigured / unavailable. The caller must `await server.connect()` before use
    (see `connect_spotify_mcp`)."""
    if not config.spotify_mcp_enabled:
        return None
    path = config.spotify_mcp_server_path
    if not path or not os.path.exists(path):
        logger.warning(
            "JARVIS_SPOTIFY_MCP is on but JARVIS_SPOTIFY_MCP_PATH is missing or not "
            "found (%r) — skipping the Spotify MCP server.", path,
        )
        return None
    try:
        from agents.mcp import MCPServerStdio
    except Exception as e:  # openai-agents built without MCP support
        logger.warning("Spotify MCP unavailable (agents.mcp import failed): %s", e)
        return None

    # The server reads/writes spotify-config.json in its own working dir; default to
    # the server repo root (two levels up from build/index.js).
    cwd = config.spotify_mcp_cwd or os.path.dirname(os.path.dirname(path)) or None
    return MCPServerStdio(
        name="spotify",
        params={"command": config.spotify_mcp_command, "args": [path], "cwd": cwd},
        cache_tools_list=True,  # tool list is stable → cache it, no round-trip per turn
        client_session_timeout_seconds=20,
    )


async def connect_spotify_mcp():
    """Build and connect the Spotify MCP server. Returns the connected server, or
    None if it's disabled/unconfigured or fails to start (JARVIS carries on either
    way). The server stays alive for the process lifetime."""
    server = build_spotify_mcp_server()
    if server is None:
        return None
    try:
        await server.connect()
        logger.info("Spotify MCP server connected (%s)", config.spotify_mcp_server_path)
        return server
    except Exception as e:  # missing node, bad path, auth not done, etc.
        logger.warning("Spotify MCP failed to start; continuing without it: %s", e)
        return None
