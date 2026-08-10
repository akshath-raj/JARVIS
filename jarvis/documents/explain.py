"""Explain a downloaded document aloud, using the local model by default."""
from __future__ import annotations

from jarvis.config import config
from jarvis.documents.parse import read_text
from jarvis.graph.llm import chat_model

_PROMPT = (
    "You are JARVIS. Read the document below and explain it to the user OUT LOUD in two or "
    "three short spoken sentences: what it is, and what it asks them to do (the key tasks, "
    "questions, or requirements). Be concrete and specific. No markdown, no lists, no preamble."
)


def explain(path: str, model_name: str | None = None) -> str:
    """Return a short spoken explanation of the document at `path`."""
    text = read_text(path)  # raises DocumentError on unreadable/empty
    model = chat_model(model_name or config.explain_model, temperature=0.2)
    resp = model.invoke([{"role": "system", "content": _PROMPT}, {"role": "user", "content": text}])
    return (getattr(resp, "content", "") or "").strip()
