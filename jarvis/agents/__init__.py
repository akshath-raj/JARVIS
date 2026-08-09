"""JARVIS multi-agent mesh: a thin router that hands off to focused specialists."""
from jarvis.agents.calendar import CalendarAgent
from jarvis.agents.chat import ChatAgent
from jarvis.agents.files import FileAgent
from jarvis.agents.music import MusicAgent
from jarvis.agents.router import RouterAgent

__all__ = ["RouterAgent", "ChatAgent", "MusicAgent", "CalendarAgent", "FileAgent"]
