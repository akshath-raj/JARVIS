"""Open a file with the right macOS app (Preview / Word / PowerPoint / VS Code).

Falls back to the system default (`open <file>`) when the mapped app isn't
installed, so it degrades gracefully on any Mac.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# extension → preferred macOS app name
_APP_BY_EXT: dict[str, str] = {
    ".pdf": "Preview",
    ".doc": "Microsoft Word",
    ".docx": "Microsoft Word",
    ".ppt": "Microsoft PowerPoint",
    ".pptx": "Microsoft PowerPoint",
    ".xls": "Microsoft Excel",
    ".xlsx": "Microsoft Excel",
    ".ipynb": "Visual Studio Code",
    ".py": "Visual Studio Code",
    ".js": "Visual Studio Code",
    ".ts": "Visual Studio Code",
    ".json": "Visual Studio Code",
    ".md": "Visual Studio Code",
}
_APP_DIRS = ("/Applications", "/System/Applications", str(Path.home() / "Applications"))


def app_installed(name: str) -> bool:
    return any(Path(d, f"{name}.app").exists() for d in _APP_DIRS)


def app_for(path: str | Path) -> str | None:
    """The preferred installed app for this file, or None to use the default."""
    app = _APP_BY_EXT.get(Path(path).suffix.lower())
    return app if app and app_installed(app) else None


def open_with_app(path: str | Path) -> str:
    """Open `path` with its preferred app (or the system default). Returns a label."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    app = app_for(p)
    cmd = ["open", "-a", app, str(p)] if app else ["open", str(p)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:  # app refused (e.g. not really installed) → default
        subprocess.run(["open", str(p)])
        return f"opened {p.name}"
    return f"opened {p.name}" + (f" in {app}" if app else "")
