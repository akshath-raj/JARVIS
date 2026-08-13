"""Force a clean RAG re-index so every document carries the full chunk metadata
(page/slide, download + modified dates, on-device location).

Existing indexes built before the metadata enrichment are re-embedded once here;
day-to-day the incremental scanner keeps things up to date without a rebuild.

    python -m scripts.reindex_rag          # or: .venv/bin/python scripts/reindex_rag.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from jarvis.config import config  # noqa: E402
from jarvis.rag import build_rag  # noqa: E402


def main() -> None:
    rag = build_rag(config)
    store = rag._store

    idx = Path(config.rag_index_dir).expanduser()
    print(f"index dir: {idx}")
    before = store.stats()
    print(f"before: {before}")

    # wipe derived index files so sync re-embeds everything with current metadata
    for name in ("vectors.npy", "chunks.json", "manifest.json"):
        p = idx / name
        if p.exists():
            p.unlink()
    rag = build_rag(config)  # reload from the now-empty index

    t = time.time()
    summary = rag.sync()  # full pass (no limit) → embeds all supported files
    dt = time.time() - t

    docs = rag.documents()
    with_dates = sum(1 for d in docs if d.get("created"))
    with_loc = sum(1 for d in docs if d.get("location"))
    print(f"done in {dt/60:.1f} min: {summary}")
    print(f"index now: {rag._store.stats()}")
    print(f"metadata coverage — dates: {with_dates}/{len(docs)}, location: {with_loc}/{len(docs)}")


if __name__ == "__main__":
    main()
