"""OpenAI Agents SDK brain for JARVIS (JARVIS_ORCHESTRATOR=openai).

A triage agent that HANDS OFF to specialist agents (Music, Browser/Tasks, Screen),
all running on a frontier OpenAI model. Feature parity with the local LangGraph
brain — same Spotify / Chrome / web / documents controllers — plus on-demand screen
understanding via an OpenAI vision model.

Imported lazily from the entrypoint so local-only installs (no `agents` package)
are unaffected.
"""
from __future__ import annotations

from jarvis.openai_agent.adapter import OpenAIAgentsLLM
from jarvis.openai_agent.brain import build_brain, describe

__all__ = ["build_brain", "describe", "OpenAIAgentsLLM"]
