"""Shared session state, handed between agents via AgentSession.userdata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from livekit.agents.llm import LLM

from jarvis.activation import WakeController
from jarvis.tools.spotify import SpotifyController


@dataclass
class JarvisContext:
    """State shared across all JARVIS agents for the life of a session.

    Long-lived, reusable resources (the Spotify controller with its cached auth
    token, the wake-word activation state machine) live here so specialist agents
    read them from `session.userdata` instead of rebuilding on handoff.
    """

    spotify: SpotifyController
    activation: WakeController
    search_mode: str = "auto"
    greeted: bool = False  # so the coordinator greets only on first entry
    # In the hybrid, device-action agents run on this local LLM (None = use the
    # session LLM, which is already local in local mode).
    local_llm: Optional[LLM] = None
