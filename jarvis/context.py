"""Shared session state, handed between agents via AgentSession.userdata."""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.activation import WakeController
from jarvis.tools.spotify import SpotifyController


@dataclass
class JarvisContext:
    """Session state shared with the agent's tools via session.userdata."""

    spotify: SpotifyController
    activation: WakeController
    search_mode: str = "auto"
    greeted: bool = False  # so JARVIS greets only once
