"""Claude desktop app backend (macOS) — experimental, opt-in.

The user prefers driving the Claude.app desktop app. Reality (verified): it's an
Electron app with no scripting dictionary, so it can only be driven via macOS
Accessibility (UI scripting) — which must be granted to this process — and even
then, reliably extracting the finished answer is hard. So this backend is OFF
unless you (a) grant Accessibility AND (b) set JARVIS_CLAUDE_DESKTOP=1. When it
can't extract a clean answer it raises BackendError, and the orchestrator falls
back to the browser backend (claude.ai). Nothing is lost by trying it.
"""
from __future__ import annotations

import os
import subprocess
import time

from jarvis.ai_apps.base import AnswerBackend, BackendError

_APP_DIRS = ("/Applications", "/System/Applications", os.path.expanduser("~/Applications"))
_ax_cache: bool | None = None


def _claude_installed() -> bool:
    return any(os.path.exists(os.path.join(d, "Claude.app")) for d in _APP_DIRS)


def accessibility_granted() -> bool:
    """True if this process may use macOS UI scripting (System Events AX)."""
    global _ax_cache
    if _ax_cache is not None:
        return _ax_cache
    try:
        p = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to tell process "Finder" to return count of windows'],
            capture_output=True, text=True, timeout=8,
        )
        _ax_cache = p.returncode == 0 and "not allowed" not in (p.stderr or "").lower()
    except Exception:
        _ax_cache = False
    return _ax_cache


def _osa(script: str, timeout: int = 20) -> str:
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise BackendError(f"AppleScript error: {p.stderr.strip()}")
    return p.stdout.strip()


class ClaudeDesktopBackend(AnswerBackend):
    name = "claude_desktop"

    def available(self) -> bool:
        return (
            _claude_installed()
            and os.getenv("JARVIS_CLAUDE_DESKTOP", "0").strip() in ("1", "true", "yes", "on")
            and accessibility_granted()
        )

    async def generate(self, assignment_path: str, prompt: str, out_format: str) -> str:
        # Best-effort UI automation: open a new chat, paste the file + prompt, submit,
        # wait, then copy the reply from the clipboard. Fragile by nature — any failure
        # raises BackendError so the orchestrator falls back to the browser backend.
        import asyncio

        return await asyncio.to_thread(self._drive, assignment_path, prompt)

    def _drive(self, assignment_path: str, prompt: str) -> str:
        path = os.path.expanduser(assignment_path)
        self._osa_activate()
        # New chat (Cmd+N), attach the file via clipboard paste, paste the prompt, send.
        self._key("n", cmd=True); time.sleep(1.0)
        self._set_clipboard_file(path); self._key("v", cmd=True); time.sleep(2.0)
        self._set_clipboard_text(prompt); self._key("v", cmd=True); time.sleep(0.5)
        self._key_return();
        answer = self._await_and_copy_reply()
        if not answer.strip():
            raise BackendError("couldn't read Claude desktop's reply")
        return answer

    # ── low-level UI helpers (System Events) ──────────────────────────────
    def _osa_activate(self) -> None:
        _osa('tell application "Claude" to activate'); time.sleep(1.2)

    def _key(self, ch: str, *, cmd: bool = False) -> None:
        mod = ' using command down' if cmd else ''
        _osa(f'tell application "System Events" to keystroke "{ch}"{mod}')

    def _key_return(self) -> None:
        _osa('tell application "System Events" to key code 36')

    def _set_clipboard_text(self, text: str) -> None:
        # Write via pbcopy to avoid AppleScript string-escaping issues.
        subprocess.run(["pbcopy"], input=text.encode(), check=True)

    def _set_clipboard_file(self, path: str) -> None:
        _osa(f'set the clipboard to (POSIX file "{path}")')

    def _await_and_copy_reply(self, settle: float = 3.0, max_wait: float = 180.0) -> str:
        # Heuristic completion: poll the clipboard after selecting+copying the reply.
        # We copy the whole conversation and return the last assistant turn's text.
        deadline = time.time() + max_wait
        last = ""
        stable = 0
        while time.time() < deadline:
            time.sleep(settle)
            self._key("a", cmd=True); self._key("c", cmd=True); time.sleep(0.3)
            cur = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
            if cur == last and cur.strip():
                stable += 1
                if stable >= 2:  # unchanged twice → generation likely done
                    return cur
            else:
                stable = 0
            last = cur
        return last
