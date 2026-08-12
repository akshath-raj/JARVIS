"""RAGPipeline — the smart, self-updating document index + question answering.

`sync()` reconciles the index with what's on disk: it re-embeds only documents
whose (size, mtime) signature changed, adds new ones, and drops ones whose files
were deleted or moved away — never re-embedding everything. `answer()` retrieves
the most relevant chunks and asks the frontier model, citing the source files.
"""
from __future__ import annotations

import logging
from pathlib import Path

from jarvis.rag.chunker import chunk_sections
from jarvis.rag.embedder import EmbedError, Embedder
from jarvis.rag.extract import SUPPORTED_EXTS, ExtractError, extract_sections
from jarvis.rag.store import VectorStore

logger = logging.getLogger("jarvis.rag")

# skip machine/system noise and our own working dirs when scanning
_SKIP_DIRS = {".git", "node_modules", ".venv", ".venv-browser", "__pycache__",
              "Library", ".Trash", ".cache", "site-packages"}
_MAX_FILE_BYTES = 40 * 1024 * 1024  # don't try to index anything over ~40MB


def _sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


class RAGPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, *, answer_model: str,
                 openai_api_key: str = "", dirs: list[str] | None = None) -> None:
        self._store = store
        self._embedder = embedder
        self._answer_model = answer_model
        self._openai_key = openai_api_key
        self._dirs = [str(Path(d).expanduser()) for d in (dirs or [])]

    # ── discovery ──────────────────────────────────────────────────────────────
    def _collect(self, dirs: list[str]) -> list[Path]:
        out: list[Path] = []
        for d in dirs:
            root = Path(d).expanduser()
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_dir():
                    continue
                if any(part in _SKIP_DIRS or part.startswith(".") for part in p.parts[len(root.parts):]):
                    continue
                if p.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                try:
                    if p.stat().st_size > _MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                out.append(p)
        return out

    # ── incremental sync ───────────────────────────────────────────────────────
    def sync(self, dirs: list[str] | None = None, *, limit: int | None = None) -> dict:
        """Reconcile the index with disk. Re-embeds only new/changed files (up to
        `limit` per call so a big backlog spreads over several cycles), prunes
        deleted ones, and caches files it can't read so they aren't retried."""
        scan_dirs = [str(Path(d).expanduser()) for d in (dirs or self._dirs)]
        files = self._collect(scan_dirs)
        current = {str(p): p for p in files}
        summary = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "failed": 0}

        for src, p in current.items():
            try:
                sig = _sig(p)
            except OSError:
                continue
            prev = self._store.doc_hash(src)
            if prev == sig or self._store.is_skipped(src, sig):
                summary["unchanged"] += 1
                continue
            if limit is not None and (summary["added"] + summary["updated"]) >= limit:
                continue  # defer the rest to the next cycle → stay responsive
            try:
                self._index_one(p, sig)
                summary["updated" if prev else "added"] += 1
            except (ExtractError, EmbedError) as e:
                logger.debug("skip %s: %s", p.name, e)
                self._store.mark_skipped(src, sig)  # remember so we don't retry it
                summary["failed"] += 1

        # documents whose files vanished (deleted or moved) — but only prune within
        # the directories we actually scanned, so a narrowed scan doesn't wipe others
        scanned_roots = tuple(scan_dirs)
        for src in list(self._store.sources()):
            if src not in current and src.startswith(scanned_roots):
                self._store.remove_document(src)
                self._store.forget(src)
                summary["removed"] += 1
        logger.debug("RAG sync: %s", summary)
        return summary

    def _index_one(self, path: Path, sig: str) -> None:
        sections = extract_sections(path)
        chunks = chunk_sections(path, sections)
        if not chunks:
            return
        vecs = self._embedder.embed([c.text for c in chunks])
        st = path.stat()
        self._store.add_document(
            str(path), hash=sig, mtime=st.st_mtime, size=st.st_size, chunks=chunks, vectors=vecs
        )

    def ingest_file(self, path: str) -> dict:
        """Index a single freshly-downloaded/created file (if supported)."""
        p = Path(path).expanduser()
        if not p.exists() or p.suffix.lower() not in SUPPORTED_EXTS:
            return {"indexed": False, "reason": "unsupported or missing"}
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                return {"indexed": False, "reason": "too large"}
            sig = _sig(p)
            if self._store.doc_hash(str(p)) == sig:
                return {"indexed": False, "reason": "already indexed"}
            self._index_one(p, sig)
            return {"indexed": True, "file": p.name}
        except (ExtractError, EmbedError) as e:
            return {"indexed": False, "reason": str(e)}

    # ── retrieval + answering ──────────────────────────────────────────────────
    def search(self, question: str, k: int = 6, *, source_contains: str = "") -> list[dict]:
        qv = self._embedder.embed_one(question)
        return self._store.search(qv, k, source_contains=source_contains)

    def answer(self, question: str, *, k: int = 6, source_contains: str = "") -> str:
        if self._store.stats()["chunks"] == 0:
            return "I haven't indexed any documents yet, sir — download some or ask me to reindex."
        hits = self.search(question, k, source_contains=source_contains)
        if not hits:
            return "I couldn't find anything relevant in your documents, sir."
        context = "\n\n".join(
            f"[{i+1}] ({h['metadata'].get('cite','?')})\n{h['text']}" for i, h in enumerate(hits)
        )
        sources = []
        for h in hits:
            c = h["metadata"].get("cite", "")
            if c and c not in sources:
                sources.append(c)
        prompt = (
            "Answer the user's question using ONLY the document excerpts below. Be "
            "concise and correct; the reply is spoken aloud so use plain prose. If the "
            "excerpts don't contain the answer, say you couldn't find it in the "
            "documents. Do not invent citations.\n\n"
            f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
        )
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model=self._answer_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            ans = (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"I couldn't reach the answering model, sir ({e})."
        top = sources[0] if sources else ""
        return ans + (f" (Source: {top}.)" if top and "couldn't find" not in ans.lower() else "")
