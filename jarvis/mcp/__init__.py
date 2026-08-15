"""JARVIS MCP server — every JARVIS capability exposed as Model Context Protocol
tools, with none of the voice/STT/agent-reasoning machinery.

Any MCP client (Claude, OpenAI, Cursor, …) can connect over streamable HTTP and
call the tools directly. See ``jarvis.mcp.server`` and ``deploy/`` for hosting.
"""
from __future__ import annotations

from jarvis.mcp.server import build_server, main

__all__ = ["build_server", "main"]
