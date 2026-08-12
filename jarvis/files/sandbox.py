"""Sandbox — confines every file operation to a root (the user's home by default).

Hard guarantees:
  * every path is resolved (symlinks + '..' collapsed) and MUST stay under the root;
    anything outside raises SandboxError.
  * there is NO delete/remove operation anywhere in this package — by construction
    JARVIS can read, copy, and move files, but never delete them.
"""
from __future__ import annotations

from pathlib import Path


class SandboxError(RuntimeError):
    pass


class Sandbox:
    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve `path` (absolute, or relative to the sandbox root) and verify it
        stays inside the sandbox. Raises SandboxError otherwise."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.root / p
        # resolve against the real filesystem where possible; for not-yet-existing
        # targets, resolve the parent and re-append the name.
        try:
            resolved = p.resolve()
        except (OSError, RuntimeError):
            resolved = (p.parent.resolve() / p.name)
        if not self._inside(resolved):
            raise SandboxError(f"'{path}' is outside your sandbox ({self.root})")
        if must_exist and not resolved.exists():
            raise SandboxError(f"no such path: {path}")
        return resolved

    def _inside(self, p: Path) -> bool:
        try:
            p.relative_to(self.root)
            return True
        except ValueError:
            return False

    def contains(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
            return True
        except SandboxError:
            return False
