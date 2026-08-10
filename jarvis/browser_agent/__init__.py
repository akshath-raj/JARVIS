"""Isolated browser agent.

browser-use pins openai/pydantic/anyio versions that conflict with the main
LiveKit + LangGraph stack, so it lives in a SEPARATE venv (.venv-browser) and
JARVIS drives it out-of-process:

  * runner.py   — runs UNDER .venv-browser; the only module that imports browser_use.
  * client.py   — runs under the main .venv; spawns the runner subprocess.
  * announcer.py— lets a background browser task speak its result via the session.

Only client.py / announcer.py are safe to import from the main app.
"""
