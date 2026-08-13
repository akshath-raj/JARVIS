"""RAGPipeline — the smart, self-updating document index + question answering.

`sync()` reconciles the index with what's on disk: it re-embeds only documents
whose (size, mtime) signature changed, adds new ones, and drops ones whose files
were deleted or moved away — never re-embedding everything. `answer()` retrieves
the most relevant chunks and asks the frontier model, citing the source files.
"""
from __future__ import annotations

import logging
import re
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


_STOP = {"the", "and", "for", "pdf", "docx", "pptx", "doc", "notes", "final",
         "draft", "copy", "document", "file", "slides", "slide", "presentation"}


def _name_overlaps(hint: str, filename: str) -> bool:
    """True if a meaningful word from `hint` (an on-screen title/description)
    appears in `filename` — used to confirm the resolved file is really the one on
    screen. If the hint carries no meaningful words, we can't disprove it → True."""
    words = {w for w in re.findall(r"[a-z0-9]+", (hint or "").lower())
             if len(w) > 2 and w not in _STOP}
    if not words:
        return True
    stem = re.sub(r"[^a-z0-9]+", " ", Path(filename).stem.lower())
    stem_words = set(stem.split())
    return bool(words & stem_words)


class RAGPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, *, answer_model: str,
                 openai_api_key: str = "", dirs: list[str] | None = None,
                 answer_base_url: str = "", answer_reasoning: str = "") -> None:
        self._store = store
        self._embedder = embedder
        self._answer_model = answer_model
        self._answer_key = openai_api_key
        # base_url of the answering LLM: Cerebras (fast) or OpenAI. Empty = OpenAI.
        self._answer_base_url = answer_base_url
        # reasoning effort for reasoning models (gpt-oss): "low" = faster. "" = omit.
        self._answer_reasoning = answer_reasoning
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

    def documents(self) -> list[dict]:
        return self._store.documents()

    def rank_documents(self, description: str, *, k: int = 12) -> list[dict]:
        """Rank indexed documents against a free-text description, best first. Each
        result is a document record with an added `score`. Combines a filename-word
        match with semantic relevance of the content, so the user never needs the
        exact filename. Used to resolve a document and to offer choices when the
        description is ambiguous."""
        desc = (description or "").strip()
        if not desc or self._store.stats()["chunks"] == 0:
            return []
        docs = self._store.documents()
        by_src = {d["source"]: d for d in docs}
        words = [w for w in re.findall(r"[a-z0-9]+", desc.lower()) if len(w) > 2]
        name_scores: dict[str, float] = {}
        for d in docs:
            stem = Path(d["source"]).stem.lower()
            hits = sum(1 for w in words if w in stem)
            if hits:
                name_scores[d["source"]] = hits / max(1, len(words))
        sem_scores: dict[str, float] = {}
        for h in self.search(desc, k):
            sem_scores[h["source"]] = sem_scores.get(h["source"], 0.0) + h.get("score", 0.0)
        combined: dict[str, float] = {}
        for src in set(name_scores) | set(sem_scores):
            combined[src] = 1.5 * name_scores.get(src, 0.0) + sem_scores.get(src, 0.0)
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for src, score in ranked:
            rec = dict(by_src.get(src, {"source": src, "filename": Path(src).name}))
            rec["score"] = round(score, 4)
            out.append(rec)
        return out

    def resolve_document(self, description: str, *, k: int = 12) -> dict | None:
        """The single best-matching document for a description (or None)."""
        ranked = self.rank_documents(description, k=k)
        return ranked[0] if ranked else None

    def locate_in_document(self, source_contains: str, text: str) -> dict | None:
        """Find where a snippet of visible text sits inside a document — the chunk
        (with its page/slide + section) that best matches it. Lets JARVIS pin the
        exact spot the user is looking at from an on-screen excerpt."""
        if not (text or "").strip():
            return None
        hits = self.search(text, 1, source_contains=source_contains)
        return hits[0] if hits else None

    def _complete(self, prompt: str, *, max_tokens: int = 900) -> str | None:
        try:
            from openai import OpenAI

            from jarvis.openai_compat import chat_complete

            client = OpenAI(api_key=self._answer_key,
                            base_url=self._answer_base_url or None)
            resp = chat_complete(
                client,
                model=self._answer_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_output=max_tokens,
                reasoning_effort=self._answer_reasoning or None,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("answer model failed: %s", e)
            return None

    def answer(self, question: str, *, k: int = 6, source_contains: str = "",
               document: str = "", page: int = 0) -> str:
        """Answer a question from the indexed documents. Optionally scope to one
        document (by free-text `document` description or `source_contains`
        substring) and/or a specific `page`/slide number."""
        if self._store.stats()["chunks"] == 0:
            return "I haven't indexed any documents yet, sir — download some or ask me to reindex."

        src_filter, label = source_contains, ""
        if document:
            doc = self.resolve_document(document)
            if not doc:
                return f"I couldn't find a document matching “{document}”, sir."
            src_filter, label = doc["source"], doc["filename"]

        if page and src_filter:
            hits = self._store.chunks_for(src_filter, page=page)
            if not hits:
                where = label or "that document"
                return f"I couldn't find page {page} in {where}, sir."
        else:
            hits = self.search(question, k, source_contains=src_filter)
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
        ans = self._complete(prompt)
        if ans is None:
            return "I couldn't reach the answering model, sir."
        top = sources[0] if sources else ""
        return ans + (f" (Source: {top}.)" if top and "couldn't find" not in ans.lower() else "")

    def summarize(self, document: str, *, page: int = 0, max_chunks: int = 40) -> str:
        """Summarise a whole document (or one page/slide) resolved from a
        free-text description."""
        if self._store.stats()["chunks"] == 0:
            return "I haven't indexed any documents yet, sir."
        doc = self.resolve_document(document) if document else None
        if not doc:
            return f"I couldn't find a document matching “{document}”, sir."
        chunks = self._store.chunks_for(doc["source"], page=page or None)
        if not chunks:
            where = f"page {page} of " if page else ""
            return f"I couldn't find {where}{doc['filename']} in the index, sir."
        body = "\n\n".join(c["text"] for c in chunks[:max_chunks])
        scope = f"page {page} of " if page else ""
        prompt = (
            f"Summarise {scope}the document below for the user, spoken aloud as plain "
            "prose (no markdown, no bullet symbols). Capture the main points and any "
            "key conclusions in a few sentences.\n\n"
            f"DOCUMENT ({doc['filename']}):\n{body}"
        )
        ans = self._complete(prompt, max_tokens=900)
        if ans is None:
            return "I couldn't reach the answering model, sir."
        return ans + f" (Source: {doc['filename']}.)"

    def explain_topic(self, question: str, *, source: str = "", document: str = "",
                      locator_text: str = "", page: int = 0, radius: int = 3) -> dict | None:
        """Explain the subtopic the user is looking at, using the indexed document
        as the source of truth. Locates the on-screen spot (from `locator_text`
        and/or `page`) inside the document, then gathers the pages just above and
        below that belong to the same subtopic — so an explanation isn't limited to
        what a single screenshot shows. Returns {answer, document, page, subtopic,
        cite}, or None if the document isn't indexed (caller should fall back to
        plain screen vision)."""
        if self._store.stats()["chunks"] == 0:
            return None
        src, label = source, (Path(source).name if source else "")
        if not src and document:
            doc = self.resolve_document(document)
            # Confidence guard: the on-screen document's own name/title should share
            # a real word with the file we resolved. If it doesn't, the live document
            # probably isn't indexed — return None so the caller falls back to plain
            # screen vision instead of confidently explaining the WRONG file.
            if doc and _name_overlaps(document, doc["filename"]):
                src, label = doc["source"], doc["filename"]
        if not src:
            return None

        center, section = None, ""
        loc = self.locate_in_document(src, locator_text) if locator_text else None
        if loc is None and page:
            pg = self._store.chunks_for(src, page=page)
            loc = pg[0] if pg else None
        if loc is not None:
            m = loc.get("metadata", {})
            center = m.get("chunk_index")
            section = m.get("section", "")
            page = page or m.get("page", 0) or 0

        if center is not None:
            chunks = self._store.context_window(src, center, radius=radius, section=section)
        elif page:
            chunks = self._store.chunks_for(src, page=page)
        else:
            chunks = self.search(question or locator_text or label, 6, source_contains=src)
        if not chunks:
            return None

        context = "\n\n".join(
            f"({c['metadata'].get('cite','?')})\n{c['text']}" for c in chunks
        )
        where = ""
        if page:
            where += f" around page {page}"
        if section:
            where += f", under “{section}”"
        prompt = (
            "The user is looking at this document on screen and wants a subtopic "
            "explained so they can understand it. Teach it clearly and simply, in "
            "your own words, using the document context below" + (where or "") + ". "
            "The context spans the pages before and after what's on screen so you "
            "have the full subtopic — use it. Spoken aloud, so plain prose, no "
            "markdown. If the context genuinely doesn't cover it, say so briefly.\n\n"
            f"DOCUMENT: {label}\nCONTEXT:\n{context}\n\n"
            f"USER: {question or 'Explain this topic to me, I am not able to understand it.'}"
        )
        ans = self._complete(prompt, max_tokens=1000)
        if ans is None:
            return None
        cite = label + (f" · p{page}" if page else "")
        return {"answer": ans, "document": label, "page": page,
                "subtopic": section, "cite": cite}
