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
    """Construct a RAGPipeline from the JARVIS config."""
    embedder = Embedder(
        backend=config.rag_embed_backend,
        model=config.rag_embed_model,
        ollama_host=config.ollama_base_url.replace("/v1", ""),
        openai_api_key=config.openai_api_key,
    )
    store = VectorStore(config.rag_index_dir, dim=config.embed_dims, embed_key=embedder.key)
    return RAGPipeline(
        store, embedder,
        answer_model=config.cloud_agent_model,
        openai_api_key=config.openai_api_key,
        dirs=config.rag_dirs,
    )
