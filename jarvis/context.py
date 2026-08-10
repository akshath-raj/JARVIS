"""Shared session state, handed to the agent's tools via AgentSession.userdata.

Two orchestrators use this:
  * langgraph (default) — only `activation`/`greeted` are needed at the LiveKit
    layer; reasoning + tools + memory live in the graph.
  * native (fallback)   — the hand-rolled agent uses `spotify`/`browser`/`tavily`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jarvis.activation import WakeController


@dataclass
class JarvisContext:
    activation: WakeController
    # native-mode controllers (unused in langgraph mode)
    spotify: object | None = None
    browser: object | None = None
    tavily: object | None = None
    # langgraph-mode memory handle (used by the background extractor wiring)
    memory: Optional[object] = None
    search_mode: str = "auto"
    greeted: bool = False  # so JARVIS greets only once
