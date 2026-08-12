"""Local vector store with incremental, self-healing sync.

Persistence (under one directory):
  * vectors.npy  — float32 (N, D) L2-normalised embeddings
  * chunks.json  — list of {id, source, text, metadata}, row-aligned to vectors
  * manifest.json— {embed_key, docs: {source: {hash, mtime, size, n_chunks}}}

The manifest's per-document content **hash** is what makes updates cheap: the
pipeline re-embeds a document only when its hash changes, drops documents whose
files have disappeared, and leaves everything else untouched. Changing the
embedding model (embed_key) invalidates the whole index so it re-embeds cleanly.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("jarvis.rag")


class VectorStore:
    def __init__(self, directory: str, *, dim: int = 768, embed_key: str = "") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vpath = self._dir / "vectors.npy"
        self._cpath = self._dir / "chunks.json"
        self._mpath = self._dir / "manifest.json"
        self._lock = threading.RLock()
        self._dim = dim
        self._embed_key = embed_key
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[dict] = []
        self._docs: dict[str, dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            manifest = json.loads(self._mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        # a change of embedding space invalidates the vectors
        if self._embed_key and manifest.get("embed_key") not in (None, self._embed_key):
            logger.info("embed model changed → resetting index")
            self._save()
            return
        self._docs = manifest.get("docs", {})
        try:
            chunks = json.loads(self._cpath.read_text(encoding="utf-8"))
            vectors = np.load(self._vpath)
            if len(chunks) == vectors.shape[0]:
                self._chunks = chunks
                self._vectors = vectors.astype(np.float32)
                if vectors.shape[0]:
                    self._dim = vectors.shape[1]
        except (OSError, ValueError, json.JSONDecodeError):
            self._chunks, self._vectors, self._docs = [], np.zeros((0, self._dim), np.float32), {}

    def _save(self) -> None:
        try:
            np.save(self._vpath, self._vectors)
            self._cpath.write_text(json.dumps(self._chunks, ensure_ascii=False), encoding="utf-8")
            self._mpath.write_text(
                json.dumps({"embed_key": self._embed_key, "docs": self._docs}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("couldn't persist index: %s", e)

    # ── manifest queries (drive incremental sync) ────────────────────────────
    def sources(self) -> set[str]:
        with self._lock:
            return set(self._docs.keys())

    def doc_hash(self, source: str) -> str | None:
        with self._lock:
            d = self._docs.get(source)
            return d.get("hash") if d else None

    def stats(self) -> dict:
        with self._lock:
            return {"documents": len(self._docs), "chunks": len(self._chunks)}

    # ── writes ────────────────────────────────────────────────────────────────
    def add_document(self, source: str, *, hash: str, mtime: float, size: int,
                     chunks: list, vectors: np.ndarray) -> None:
        """Add/replace a document's chunks. `chunks` are Chunk objects, `vectors`
        an (n, D) array aligned to them."""
        with self._lock:
            self._remove_rows(source)  # replace if it already existed
            rows = []
            for i, ch in enumerate(chunks):
                rows.append({
                    "id": f"{Path(source).name}#{i}",
                    "source": source,
                    "text": ch.text,
                    "metadata": ch.metadata,
                })
            if rows:
                self._chunks.extend(rows)
                self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)]) \
                    if self._vectors.size else vectors.astype(np.float32)
                self._dim = self._vectors.shape[1]
            self._docs[source] = {"hash": hash, "mtime": mtime, "size": size, "n_chunks": len(rows)}
            self._save()

    def remove_document(self, source: str) -> None:
        with self._lock:
            self._remove_rows(source)
            self._docs.pop(source, None)
            self._save()

    def _remove_rows(self, source: str) -> None:
        if not self._chunks:
            return
        keep = [i for i, c in enumerate(self._chunks) if c["source"] != source]
        if len(keep) != len(self._chunks):
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self._dim), np.float32)

    # ── search ────────────────────────────────────────────────────────────────
    def search(self, query_vec: np.ndarray, k: int = 6, *, source_contains: str = "") -> list[dict]:
        """Return the top-k chunks (each with an added 'score') by cosine similarity.
        Vectors are pre-normalised, so a dot product is the cosine."""
        with self._lock:
            if not self._chunks:
                return []
            idxs = list(range(len(self._chunks)))
            if source_contains:
                needle = source_contains.lower()
                idxs = [i for i in idxs if needle in self._chunks[i]["source"].lower()]
                if not idxs:
                    return []
            sub = self._vectors[idxs]
            sims = sub @ query_vec.astype(np.float32)
            order = np.argsort(-sims)[:k]
            out = []
            for j in order:
                i = idxs[int(j)]
                c = dict(self._chunks[i])
                c["score"] = float(sims[int(j)])
                out.append(c)
            return out
