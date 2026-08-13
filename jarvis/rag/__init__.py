"""Document RAG: local embeddings + a self-updating vector store + answering.

Build the pieces with `build_rag()`; the pipeline embeds documents locally
(Ollama), indexes them incrementally, and answers questions by retrieving the
most relevant chunks and asking the frontier model (citing sources).
"""
from __future__ import annotations

from jarvis.rag.embedder import Embedder
from jarvis.rag.pipeline import RAGPipeline
from jarvis.rag.store import VectorStore

__all__ = ["Embedder", "RAGPipeline", "VectorStore", "build_rag"]


def build_rag(config) -> RAGPipeline:
    """Construct a RAGPipeline from the JARVIS config.

    Retrieval embeddings stay local (Ollama) — Cerebras has no embeddings endpoint —
    but the ANSWER is generated on Cerebras (gpt-oss-120b) when agent_provider is
    "cerebras" and a key is set, for fast, uniform responses. Falls back to the
    OpenAI answer model only if the Cerebras key is missing.
    """
    embedder = Embedder(
        backend=config.rag_embed_backend,
        model=config.rag_embed_model,
        ollama_host=config.ollama_base_url.replace("/v1", ""),
        openai_api_key=config.openai_api_key,
    )
    store = VectorStore(config.rag_index_dir, dim=config.embed_dims, embed_key=embedder.key)

    use_cerebras = config.agent_provider == "cerebras" and bool(config.cerebras_api_key)
    if use_cerebras:
        answer_model = config.cerebras_model
        answer_key = config.cerebras_api_key
        answer_base_url = config.cerebras_base_url
        answer_reasoning = "low"     # gpt-oss reasons; keep it minimal for speed
    else:
        answer_model = config.cloud_agent_model
        answer_key = config.openai_api_key
        answer_base_url = ""
        answer_reasoning = ""

    return RAGPipeline(
        store, embedder,
        answer_model=answer_model,
        openai_api_key=answer_key,
        answer_base_url=answer_base_url,
        answer_reasoning=answer_reasoning,
        dirs=config.rag_dirs,
    )
