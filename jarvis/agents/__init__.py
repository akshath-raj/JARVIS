"""JARVIS multi-agent mesh: a thin router that hands off to focused specialists.

Sandboxed by design: the only capability that touches the machine is Spotify
playback control. There are no file, shell, calendar, or delete/remove tools.
"""
from jarvis.agents.chat import ChatAgent
from jarvis.agents.music import MusicAgent
from jarvis.agents.router import RouterAgent

__all__ = ["RouterAgent", "ChatAgent", "MusicAgent"]
