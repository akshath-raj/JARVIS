"""JARVIS agent.

`JarvisAgent` is the master coordinator (supervisor pattern). It answers general
questions and controls Spotify with its own inline music tools (unchanged), and it
DELEGATES browser/web requests to focused specialists via the `browser` and
`web_search` tools. There is no LiveKit agent handoff/swap — an earlier handoff
mesh was removed because local models emitted tool calls as text after a swap
instead of executing them. Each specialist instead runs its own scoped
tool-calling loop, which executes reliably. Add a new domain by writing a
Specialist and registering one delegation tool here.

Capabilities that touch the machine: Spotify (background) and Chrome (foreground).
No file, shell, calendar, or delete/remove tools.
"""
from jarvis.agents.jarvis_agent import JarvisAgent

__all__ = ["JarvisAgent"]
