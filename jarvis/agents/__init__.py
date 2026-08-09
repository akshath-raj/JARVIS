"""JARVIS agent.

A single agent answers general questions and controls Spotify. (An earlier
router+specialist handoff mesh was removed: agent handoffs were unreliable with
local models — the model emitted tool calls as text after a handoff instead of
executing them.)

Sandboxed: the only capability that touches the machine is Spotify. No file,
shell, calendar, or delete/remove tools.
"""
from jarvis.agents.jarvis_agent import JarvisAgent

__all__ = ["JarvisAgent"]
