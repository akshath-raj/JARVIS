"""Local model factories for the LangGraph brain (Ollama, no cloud).

`langchain-ollama` talks to Ollama's native API, so it wants the base URL WITHOUT
the `/v1` OpenAI-compat suffix that the rest of the app uses. We normalise here.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from jarvis.config import config


def _ollama_host() -> str:
    # config.ollama_base_url is the OpenAI-compat endpoint (…:11434/v1); ChatOllama
    # wants the native host (…:11434).
    return config.ollama_base_url.rstrip("/").removesuffix("/v1").rstrip("/")


def chat_model(model: str | None = None, *, temperature: float = 0.2) -> ChatOllama:
    """A local Ollama chat model for the graph (tool-calling capable)."""
    return ChatOllama(
        model=model or config.ollama_model,
        base_url=_ollama_host(),
        temperature=temperature,
    )


def embeddings() -> OllamaEmbeddings:
    """Local Ollama embeddings for the long-term memory store."""
    return OllamaEmbeddings(model=config.embed_model, base_url=_ollama_host())
