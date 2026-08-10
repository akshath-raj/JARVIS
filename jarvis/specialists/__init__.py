"""Focused sub-agents ("specialists") the supervisor delegates to.

Each specialist is a lean prompt + scoped tools + its own local tool-calling loop
(see base.py). Music is intentionally NOT a specialist — it stays as reliable
inline tools on the supervisor. Add a new domain by writing a Specialist subclass
here and registering one delegation tool on the supervisor agent.
"""
from jarvis.specialists.base import Specialist, ToolSpec
from jarvis.specialists.browser import BrowserSpecialist

__all__ = ["Specialist", "ToolSpec", "BrowserSpecialist"]
