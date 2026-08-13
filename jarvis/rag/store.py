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
        self._skips: dict[str, str] = {}  # files that couldn't be indexed → don't retry
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
        self._skips = manifest.get("skips", {})
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
                json.dumps(
                    {"embed_key": self._embed_key, "docs": self._docs, "skips": self._skips},
                    ensure_ascii=False, indent=2,
                ),
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

    def is_skipped(self, source: str, sig: str) -> bool:
        """True if this exact file version already failed to index (don't retry)."""
        with self._lock:
            return self._skips.get(source) == sig

    def mark_skipped(self, source: str, sig: str) -> None:
        with self._lock:
            self._skips[source] = sig
            self._save()

    def forget(self, source: str) -> None:
        """Drop any skip record for a file that has vanished."""
        with self._lock:
            if source in self._skips:
                self._skips.pop(source, None)
                self._save()

    def stats(self) -> dict:
        with self._lock:
            return {"documents": len(self._docs), "chunks": len(self._chunks)}

    def documents(self) -> list[dict]:
        """One record per indexed document with its file metadata (filename,
        location on device, created/downloaded + modified dates, chunk count).
        Used to resolve a document from a description and to answer "where/when"
        questions about a file."""
        with self._lock:
            seen: dict[str, dict] = {}
            for c in self._chunks:
                src = c["source"]
                if src in seen:
                    continue
                m = c.get("metadata", {})
                d = self._docs.get(src, {})
                seen[src] = {
                    "source": src,
                    "filename": m.get("filename", Path(src).name),
                    "location": m.get("location", str(Path(src).parent)),
                    "created": m.get("created", ""),
                    "modified": m.get("modified", ""),
                    "ext": m.get("ext", Path(src).suffix.lstrip(".")),
                    "n_chunks": d.get("n_chunks", 0),
                }
            return list(seen.values())

    def context_window(self, source_contains: str, center_index: int, *,
                       radius: int = 2, section: str = "") -> list[dict]:
        """Chunks around a located spot in a document: those within
        `center_index ± radius` (spanning the pages just above and below), plus any
        sharing the same `section` title. This assembles a whole subtopic even when
        it runs across several pages, ordered as in the document."""
        with self._lock:
            needle = source_contains.lower()
            lo, hi = center_index - radius, center_index + radius
            out = []
            for c in self._chunks:
                if needle and needle not in c["source"].lower():
                    continue
                m = c.get("metadata", {})
                idx = m.get("chunk_index")
                in_window = idx is not None and lo <= idx <= hi
                same_section = bool(section) and m.get("section") == section
                if in_window or same_section:
                    out.append(dict(c))
            out.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
            return out

    def chunks_for(self, source_contains: str, *, page: int | None = None) -> list[dict]:
        """All chunks of the document(s) whose path contains `source_contains`,
        optionally only those on a given page/slide, ordered as in the document.
        Powers "explain page 5 / slide 3" and whole-document summaries."""
        with self._lock:
            needle = source_contains.lower()
            out = []
            for c in self._chunks:
                if needle and needle not in c["source"].lower():
                    continue
                if page is not None and c.get("metadata", {}).get("page") != page:
                    continue
                out.append(dict(c))
            out.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
            return out

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
