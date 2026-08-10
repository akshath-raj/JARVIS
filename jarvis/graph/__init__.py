"""The JARVIS LangGraph brain: a single react-agent (all tools) personalised by a
persistent local memory store, wrapped for the LiveKit voice pipeline.

Public surface:
  * build_graph() -> (compiled_graph, memory)
  * VoiceLLMAdapter — LiveKit LLM that speaks only the model's final reply
  * describe() — one-line summary for logs
"""
from jarvis.graph.adapter import VoiceLLMAdapter
from jarvis.graph.build import build_graph, describe

__all__ = ["build_graph", "describe", "VoiceLLMAdapter"]
