"""Text embedder for the document RAG index.

Default backend is **local Ollama** (`nomic-embed-text`, 768-dim) so the document
corpus never leaves the machine — only the small set of chunks retrieved for a
question is later sent to the answering model. An OpenAI backend is available for
users who prefer it (`JARVIS_RAG_EMBED=openai`).
"""
from __future__ import annotations

import logging

import numpy as np
import requests

logger = logging.getLogger("jarvis.rag")


class EmbedError(RuntimeError):
    pass


class Embedder:
    def __init__(
        self,
        *,
        backend: str = "ollama",
        model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
        openai_api_key: str = "",
        batch: int = 64,
    ) -> None:
        self._backend = backend
        self._model = model
        self._host = ollama_host.rstrip("/")
        self._openai_key = openai_api_key
        self._batch = batch
        self._dim: int | None = None

    @property
    def key(self) -> str:
        """A stable identifier for the embedding space; changing it invalidates the
        index (so a re-embed is forced when the model/backend changes)."""
        return f"{self._backend}:{self._model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised embeddings."""
        if not texts:
            return np.zeros((0, self._dim or 768), dtype=np.float32)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            chunk = texts[i : i + self._batch]
            vecs.extend(self._ollama(chunk) if self._backend == "ollama" else self._openai(chunk))
        arr = np.asarray(vecs, dtype=np.float32)
        # normalise so a dot product is cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._dim = arr.shape[1]
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _ollama(self, texts: list[str]) -> list[list[float]]:
        try:
            r = requests.post(
                f"{self._host}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=120,
            )
            r.raise_for_status()
            embs = r.json().get("embeddings")
            if not embs:
                raise EmbedError("empty embedding response")
            return embs
        except requests.RequestException as e:
            raise EmbedError(
                f"local embedding failed ({e}); is Ollama running and "
                f"'{self._model}' pulled? (ollama pull {self._model})"
            ) from e

    def _openai(self, texts: list[str]) -> list[list[float]]:
        if not self._openai_key:
            raise EmbedError("OpenAI embedding backend needs OPENAI_API_KEY")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._openai_key)
            resp = client.embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            raise EmbedError(f"OpenAI embedding failed: {e}") from e
