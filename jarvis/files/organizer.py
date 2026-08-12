"""FileOrganizer — read / copy / move / open files, and tidy cluttered folders.

Everything is confined to the Sandbox (the user's home). It can COPY or MOVE
files and open them, and it can reorganise a folder into category subfolders,
consolidating exact duplicates into a `_Duplicates` folder. It NEVER deletes:
duplicates are moved aside, not removed, and "move" preserves the file at its new
location. Collisions are auto-renamed ("file (1).pdf") so nothing is overwritten.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from jarvis.files.sandbox import Sandbox, SandboxError

logger = logging.getLogger("jarvis.files")

CATEGORIES: dict[str, set[str]] = {
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages", ".tex"},
    "Presentations": {".ppt", ".pptx", ".key"},
    "Spreadsheets": {".xls", ".xlsx", ".csv", ".numbers"},
    "Images": {".png", ".jpg", ".jpeg", ".gif", ".heic", ".webp", ".svg", ".bmp", ".tiff"},
    "Audio": {".mp3", ".wav", ".aac", ".flac", ".m4a", ".aiff"},
    "Video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"},
    "Archives": {".zip", ".tar", ".gz", ".rar", ".7z", ".dmg"},
    "Code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".rb", ".sh", ".json", ".html", ".css", ".ipynb"},
    "Installers": {".pkg", ".app", ".exe", ".deb"},
}
_MANAGED = set(CATEGORIES) | {"_Duplicates"}


def _category(ext: str) -> str:
    ext = ext.lower()
    for name, exts in CATEGORIES.items():
        if ext in exts:
            return name
    return "Other"


class FileError(RuntimeError):
    pass


class FileOrganizer:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sb = sandbox

    # ── helpers ────────────────────────────────────────────────────────────────
    def _unique(self, dest: Path) -> Path:
        """Never overwrite: if dest exists, append ' (1)', ' (2)', …"""
        if not dest.exists():
            return dest
        stem, suffix, parent = dest.stem, dest.suffix, dest.parent
        i = 1
        while True:
            cand = parent / f"{stem} ({i}){suffix}"
            if not cand.exists():
                return cand
            i += 1

    @staticmethod
    def _hash(p: Path, limit: int = 8 * 1024 * 1024) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
                if f.tell() >= limit:
                    break
        return h.hexdigest()

    # ── explicit copy / move (sandbox-checked, never overwrites) ───────────────
    def copy(self, src: str, dst: str) -> str:
        s = self.sb.resolve(src, must_exist=True)
        d = self._dest_path(s, dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        d = self._unique(d)
        if s.is_dir():
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
        return f"Copied {s.name} to {d.parent}."

    def move(self, src: str, dst: str) -> str:
        s = self.sb.resolve(src, must_exist=True)
        d = self._dest_path(s, dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        d = self._unique(d)
        shutil.move(str(s), str(d))
        return f"Moved {s.name} to {d.parent}."

    def _dest_path(self, src: Path, dst: str) -> Path:
        """If dst is (or looks like) a directory, keep the source filename inside it;
        otherwise treat dst as the full target path."""
        d = self.sb.resolve(dst)
        if d.is_dir() or dst.endswith("/") or not d.suffix:
            return d / src.name
        return d

    # ── open ────────────────────────────────────────────────────────────────────
    def open_paths(self, paths: list[str], *, cap: int = 12) -> str:
        opened = []
        for path in paths[:cap]:
            p = self.sb.resolve(path, must_exist=True)
            subprocess.Popen(["open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            opened.append(p.name)
        if not opened:
            return "There was nothing to open, sir."
        extra = f" (+{len(paths) - cap} more)" if len(paths) > cap else ""
        return f"Opened {len(opened)} item(s): " + ", ".join(opened) + extra

    def recent_files(self, folder: str, *, n: int = 5, exts: set[str] | None = None) -> list[Path]:
        root = self.sb.resolve(folder, must_exist=True)
        files = [p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")]
        if exts:
            files = [p for p in files if p.suffix.lower() in exts]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:n]

    def list_dir(self, folder: str) -> list[Path]:
        root = self.sb.resolve(folder, must_exist=True)
        return sorted([p for p in root.iterdir() if not p.name.startswith(".")])

    # ── reorganise a folder ─────────────────────────────────────────────────────
    def organize(self, folder: str, *, mode: str = "move", dedupe: bool = True) -> dict:
        """Sort a folder's loose files into category subfolders. Exact duplicates are
        moved into `_Duplicates` (kept, never deleted). Returns a summary."""
        root = self.sb.resolve(folder, must_exist=True)
        if not root.is_dir():
            raise FileError(f"{root} is not a folder")
        act = self.copy if mode == "copy" else self.move
        summary = {"moved": 0, "duplicates": 0, "skipped": 0, "by_category": {}}
        seen: dict[str, Path] = {}

        for p in list(root.iterdir()):
            if p.is_dir() or p.name.startswith("."):
                continue
            if p.parent.name in _MANAGED:  # already sorted
                continue
            try:
                if dedupe:
                    digest = self._hash(p)
                    if digest in seen:
                        act(str(p), str(root / "_Duplicates") + "/")
                        summary["duplicates"] += 1
                        continue
                    seen[digest] = p
                cat = _category(p.suffix)
                act(str(p), str(root / cat) + "/")
                summary["moved"] += 1
                summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            except (OSError, SandboxError, FileError) as e:
                logger.info("organize skip %s: %s", p.name, e)
                summary["skipped"] += 1
        return summary
