"""AnswerBackend — pluggable "get an AI to do the assignment" providers.

A backend takes the assignment file + a strong prompt + the desired output format
and returns the answer CONTENT (markdown/code); JARVIS then assembles the file.
The registry picks the first *available* backend in the user's preferred order.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BackendError(RuntimeError):
    pass


class AnswerBackend(ABC):
    name: str = "backend"

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now (installed, permitted, keyed)."""

    @abstractmethod
    async def generate(self, assignment_path: str, prompt: str, out_format: str) -> str:
        """Return the answer content for the assignment. Raises BackendError."""
