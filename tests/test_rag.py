"""Tests for the document RAG: extraction, chunking, and the self-updating store.
Offline — a deterministic fake embedder stands in for Ollama/OpenAI."""
from __future__ import annotations

import hashlib

import numpy as np

from jarvis.rag import RAGPipeline, VectorStore
from jarvis.rag.chunker import chunk_sections
from jarvis.rag.extract import Section, extract_sections


class FakeEmbedder:
    key = "fake:test"

    def embed(self, texts):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8)[:8].astype(np.float32)
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 8), np.float32)

    def embed_one(self, t):
        return self.embed([t])[0]


# ── extraction / chunking ────────────────────────────────────────────────────
def test_extract_markdown_sections(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Alpha\nfirst body\n\n# Beta\nsecond body")
    secs = extract_sections(f)
    assert [s.title for s in secs] == ["Alpha", "Beta"]


def test_chunker_attaches_metadata(tmp_path):
    f = tmp_path / "n.md"
    secs = [Section(title="Intro", text="hello world. " * 200, page=None)]
    chunks = chunk_sections(f, secs, max_chars=300)
    assert len(chunks) > 1               # long section split into several chunks
    m = chunks[0].metadata
    assert m["filename"] == "n.md" and m["section"] == "Intro" and "cite" in m
    assert m["chunk_index"] == 0


def test_pptx_pages_and_docx_headings(tmp_path):
    from docx import Document
    from pptx import Presentation

    d = Document(); d.add_heading("H1", level=1); d.add_paragraph("body")
    d.save(str(tmp_path / "d.docx"))
    assert any(s.title == "H1" for s in extract_sections(tmp_path / "d.docx"))

    prs = Presentation(); s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Slide A"; s.placeholders[1].text = "content"
    prs.save(str(tmp_path / "p.pptx"))
    secs = extract_sections(tmp_path / "p.pptx")
    assert secs[0].page == 1


# ── vector store ─────────────────────────────────────────────────────────────
def test_store_add_search_remove(tmp_path):
    emb = FakeEmbedder()
    store = VectorStore(str(tmp_path / "idx"), dim=8, embed_key=emb.key)

    class C:
        def __init__(self, text):
            self.text = text
            self.metadata = {"source": "a.txt", "cite": "a.txt"}

    chunks = [C("apple pie recipe"), C("rocket engine thrust")]
    store.add_document("a.txt", hash="h1", mtime=1.0, size=10,
                       chunks=chunks, vectors=emb.embed([c.text for c in chunks]))
    assert store.stats() == {"documents": 1, "chunks": 2}
    hits = store.search(emb.embed_one("apple pie recipe"), k=1)
    assert hits and hits[0]["text"] == "apple pie recipe"
    store.remove_document("a.txt")
    assert store.stats() == {"documents": 0, "chunks": 0}


def test_store_persists_and_reloads(tmp_path):
    emb = FakeEmbedder()
    d = str(tmp_path / "idx")

    class C:
        text = "persisted chunk"
        metadata = {"source": "a.txt", "cite": "a.txt"}

    VectorStore(d, dim=8, embed_key=emb.key).add_document(
        "a.txt", hash="h", mtime=1.0, size=1, chunks=[C()], vectors=emb.embed(["persisted chunk"]))
    reloaded = VectorStore(d, dim=8, embed_key=emb.key)
    assert reloaded.stats()["chunks"] == 1


def test_embed_key_change_invalidates_index(tmp_path):
    d = str(tmp_path / "idx")

    class C:
        text = "x"
        metadata = {"source": "a.txt", "cite": "a"}

    VectorStore(d, dim=8, embed_key="fake:v1").add_document(
        "a.txt", hash="h", mtime=1, size=1, chunks=[C()], vectors=FakeEmbedder().embed(["x"]))
    # a different embedding model must reset the index rather than mix spaces
    assert VectorStore(d, dim=8, embed_key="fake:v2").stats()["chunks"] == 0


# ── incremental pipeline sync (add / update / delete) ────────────────────────
def _pipeline(tmp_path, docs):
    store = VectorStore(str(tmp_path / "idx"), dim=8, embed_key="fake:test")
    return RAGPipeline(store, FakeEmbedder(), answer_model="x", dirs=[str(docs)]), store


def test_incremental_sync_add_update_delete(tmp_path):
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "a.txt").write_text("alpha content")
    (docs / "b.md").write_text("# H\nbeta content")
    rag, store = _pipeline(tmp_path, docs)

    s1 = rag.sync()
    assert s1["added"] == 2 and store.stats()["documents"] == 2

    # no changes → nothing re-embedded
    assert rag.sync() == {"added": 0, "updated": 0, "removed": 0, "unchanged": 2, "failed": 0}

    # edit one file → exactly one update
    import os, time
    (docs / "a.txt").write_text("alpha content changed substantially")
    os.utime(docs / "a.txt", (time.time() + 5, time.time() + 5))
    s3 = rag.sync()
    assert s3["updated"] == 1 and s3["unchanged"] == 1

    # delete a file → auto-removed from the index
    (docs / "b.md").unlink()
    s4 = rag.sync()
    assert s4["removed"] == 1 and store.stats()["documents"] == 1


def test_failed_extraction_is_cached_not_retried(tmp_path):
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "scanned.txt").write_text("")  # no extractable text → ExtractError
    rag, store = _pipeline(tmp_path, docs)
    s1 = rag.sync()
    assert s1["failed"] == 1
    # the failure is remembered, so the next scan skips it instead of re-trying
    s2 = rag.sync()
    assert s2["failed"] == 0 and s2["unchanged"] == 1


def test_scan_limit_spreads_work(tmp_path):
    docs = tmp_path / "docs"; docs.mkdir()
    for i in range(5):
        (docs / f"f{i}.txt").write_text(f"document number {i}")
    rag, store = _pipeline(tmp_path, docs)
    s = rag.sync(limit=2)          # only index 2 this cycle
    assert s["added"] == 2
    assert rag.sync(limit=2)["added"] == 2   # next cycle picks up 2 more
    assert rag.sync(limit=2)["added"] == 1   # and the last one


def test_ingest_single_file(tmp_path):
    docs = tmp_path / "docs"; docs.mkdir()
    rag, store = _pipeline(tmp_path, docs)
    f = docs / "new.txt"; f.write_text("freshly downloaded report")
    assert rag.ingest_file(str(f))["indexed"] is True
    assert store.stats()["documents"] == 1
    # second ingest is a no-op (already indexed, unchanged)
    assert rag.ingest_file(str(f))["indexed"] is False
