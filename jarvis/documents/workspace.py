"""Per-session workspace: what assignment is loaded and what answer we produced.

Lets the conversation flow across turns — "download & explain it" sets the current
assignment; "finish it" reads that assignment; "open it" opens the produced answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md"}


@dataclass
class Workspace:
    downloads_dir: str
    current_assignment: str | None = None
    explanation: str | None = None
    answer_file: str | None = None

    def newest_document(self, since: float | None = None) -> str | None:
        """Newest document file in the downloads dir (optionally modified after `since`)."""
        d = Path(self.downloads_dir).expanduser()
        if not d.exists():
            return None
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _DOC_EXTS]
        if since is not None:
            files = [p for p in files if p.stat().st_mtime > since]
        if not files:
            return None
        return str(max(files, key=lambda p: p.stat().st_mtime))

    def set_assignment(self, path: str, explanation: str | None = None) -> None:
        self.current_assignment = path
        self.explanation = explanation
        self.answer_file = None  # a new assignment invalidates the old answer

    def set_answer(self, path: str) -> None:
        self.answer_file = path
