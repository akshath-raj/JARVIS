"""Shared session state, handed between agents via AgentSession.userdata."""
from __future__ import annotations

from dataclasses import dataclass

from jarvis.tools.spotify import SpotifyController


@dataclass
class JarvisContext:
    """State shared across all JARVIS agents for the life of a session.

    Kept deliberately small: long-lived, reusable resources (like the Spotify
    controller with its cached auth token) live here so specialist agents read
    them from `session.userdata` instead of re-instantiating on every handoff.
    """

    spotify: SpotifyController
    search_mode: str = "auto"
