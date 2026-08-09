"""Shared session state, handed between agents via AgentSession.userdata."""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.tools.spotify import SpotifyController
from jarvis.wake import WakeGate


@dataclass
class JarvisContext:
    """State shared across all JARVIS agents for the life of a session.

    Long-lived, reusable resources (the Spotify controller with its cached auth
    token, the wake-word gate with its loaded model) live here so specialist
    agents read them from `session.userdata` instead of rebuilding on handoff.
    """

    spotify: SpotifyController
    wake: WakeGate
    search_mode: str = "auto"
    greeted: bool = False  # so the coordinator greets only on first entry
