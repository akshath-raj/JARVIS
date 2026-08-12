"""Section-aware chunking with metadata.

Each section is split into ~`max_chars` windows on sentence/paragraph boundaries
with a small overlap, so a chunk rarely cuts mid-thought. Every chunk carries
metadata (source path, filename, type, section title, page/slide) used both for
retrieval filtering and for citing the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.rag.extract import Section


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_units(text: str) -> list[str]:
    # paragraphs first, then long paragraphs into sentences
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= 1200:
            units.append(para)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units


def chunk_sections(
    path: str | Path, sections: list[Section], *, max_chars: int = 1100, overlap: int = 180
) -> list[Chunk]:
    p = Path(path)
    base = {
        "source": str(p),
        "filename": p.name,
        "ext": p.suffix.lower().lstrip("."),
    }
    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        units = _split_units(sec.text)
        buf = ""
        for unit in units:
            if buf and len(buf) + len(unit) + 1 > max_chars:
                chunks.append(_mk(buf, base, sec, idx))
                idx += 1
                # carry a tail of the previous chunk for context continuity
                buf = (buf[-overlap:] + " " + unit).strip() if overlap else unit
            else:
                buf = f"{buf} {unit}".strip()
        if buf:
            chunks.append(_mk(buf, base, sec, idx))
            idx += 1
    return chunks


def _mk(text: str, base: dict, sec: Section, idx: int) -> Chunk:
    meta = dict(base)
    meta["section"] = sec.title
    if sec.page is not None:
        meta["page"] = sec.page
    meta["chunk_index"] = idx
    # a compact human-readable citation label
    loc = f"p{sec.page}" if sec.page is not None else sec.title
    meta["cite"] = f"{base['filename']} · {loc}"
    return Chunk(text=text, metadata=meta)
