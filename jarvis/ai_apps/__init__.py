"""Pluggable AI backends that complete an assignment (Claude desktop → browser)."""
from jarvis.ai_apps.base import AnswerBackend, BackendError
from jarvis.ai_apps.registry import available_backends

__all__ = ["AnswerBackend", "BackendError", "available_backends"]
