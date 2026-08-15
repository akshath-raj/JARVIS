"""Full control of Apple's **Pages** word processor, plus the system clipboard.

Pages exposes only a thin AppleScript surface (create a document, read/replace its
whole body text, save/export). Everything richer — inserting at the cursor,
selecting, headings/titles, tables, and images — has no scripting API, so we drive
it the way a person would: through the **clipboard** and **UI scripting** (System
Events keystrokes and menu clicks). That combination gives JARVIS genuine full
control of the app:

* Document — new document, read the text, replace/append text, save, close (AppleScript).
* Clipboard — get/set the pasteboard, and copy / cut / paste / select-all / delete
  the current selection (the copy-paste ability the user asked for).
* Rich insert — headings & titles paste as styled RTF (real bold/large text), plain
  text pastes as-is, all AT THE CURSOR so nothing already typed is lost.
* Tables — inserted via Pages' Insert ▸ Table menu.
* Images — an image file is put on the clipboard as a picture and pasted in.

AppleScript ▸ Pages needs Automation permission; the keystroke/menu parts need
Accessibility permission (System Settings ▸ Privacy). When either is denied macOS
raises an error, which we surface as a spoken-friendly `PagesError`.
"""
from __future__ import annotations

import subprocess
import tempfile
import time

# Font sizes (points) for the paragraph "styles" we synthesise via RTF. Pages has no
# scriptable paragraph styles, so a title/heading is just large bold text — visually
# a heading, and editable like any other text.
_STYLE_SIZES = {"title": 56, "subtitle": 32, "heading": 40, "subheading": 30, "body": 22}


class PagesError(RuntimeError):
    pass


def _osa(script: str, timeout: int = 25) -> str:
    p = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
    )
    if p.returncode != 0:
        err = p.stderr.strip()
        if "-1743" in err or "not allowed" in err.lower() or "assistive" in err.lower():
            raise PagesError(
                "I need permission to control Pages, sir — enable JARVIS under "
                "Privacy & Security ▸ Automation and ▸ Accessibility in System Settings."
            )
        raise PagesError(err or "Pages AppleScript failed")
    return p.stdout.strip()


def _esc(text: str) -> str:
    """Escape a Python string for embedding inside an AppleScript double-quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _rtf_escape(text: str) -> str:
    """Escape text for an RTF body: backslash/braces, and newlines as \\par."""
    out = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return out.replace("\r\n", "\\par ").replace("\n", "\\par ").replace("\r", "\\par ")


class PagesController:
    """Drive Pages through AppleScript + the clipboard + UI scripting."""

    def __init__(self, app_name: str = "Pages") -> None:
        self.app_name = app_name

    # ── low-level clipboard (plain text) ─────────────────────────────────
    def get_clipboard(self) -> str:
        """Return the current plain-text clipboard contents (empty if not text)."""
        p = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=10)
        return p.stdout

    def set_clipboard(self, text: str) -> None:
        """Put plain text on the system clipboard (so it can be pasted anywhere)."""
        subprocess.run(["pbcopy"], input=text, text=True, timeout=10)

    # ── UI scripting helpers (act on the frontmost Pages window) ─────────
    def _activate(self) -> None:
        """Launch (if needed) and bring Pages to the front via LaunchServices.

        We deliberately use `open -a` rather than AppleScript `activate`: in
        sandboxed/headless contexts the Apple-Event `activate` can fail with -600
        ("Application isn't running") before the app is up, whereas `open -a`
        launches reliably and also focuses the window."""
        try:
            subprocess.run(
                ["open", "-a", self.app_name], capture_output=True, text=True,
                timeout=15, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise PagesError(f"couldn't launch {self.app_name}") from e
        # give a cold launch a moment; then confirm it's actually running
        for _ in range(20):
            if self._is_running():
                return
            time.sleep(0.3)

    def _is_running(self) -> bool:
        out = _osa(
            'tell application "System Events" to (name of processes) contains '
            f'"{_esc(self.app_name)}"'
        )
        return out.strip().lower() == "true"

    def _keystroke(self, key: str, *, using: str = "") -> None:
        """Send a Cmd-combo keystroke to Pages (e.g. key='v', using='command down')."""
        clause = f" using {{{using}}}" if using else ""
        _osa(
            'tell application "System Events" to tell process "Pages" '
            f'to keystroke "{_esc(key)}"{clause}'
        )

    def _paste(self) -> None:
        self._keystroke("v", using="command down")

    # ── document lifecycle ───────────────────────────────────────────────
    def new_document(self) -> str:
        """Open Pages and create a fresh blank document. Returns a confirmation."""
        self._activate()  # launch via LaunchServices + wait until it's running
        _osa('tell application "Pages" to make new document')
        return "new Pages document ready"

    def get_text(self) -> str:
        """Read the front document's whole body text."""
        return _osa(
            'tell application "Pages" to get body text of front document'
        )

    def set_text(self, text: str) -> None:
        """Replace the front document's ENTIRE body text (used for clear/overwrite)."""
        _osa(
            'tell application "Pages" to set body text of front document '
            f'to "{_esc(text)}"'
        )

    def append_text(self, text: str) -> None:
        """Append to the end of the document body (AppleScript, keeps existing text)."""
        _osa(
            'tell application "Pages"\n'
            "  set d to front document\n"
            f'  set body text of d to (body text of d) & "{_esc(text)}"\n'
            "end tell"
        )

    def clear(self) -> str:
        """Erase all text in the front document."""
        self.set_text("")
        return "cleared the document"

    def save(self, path: str = "") -> str:
        """Save the front document. Pass a `path` to save a NEW document to a specific
        file (no Save dialog); omit it to re-save an already-saved document. A bare
        `save` on an unsaved doc would pop a modal dialog and hang, so callers that
        just created a document should always pass a path."""
        if path:
            from pathlib import Path
            p = Path(path).expanduser()
            if p.suffix.lower() != ".pages":
                p = p.with_suffix(".pages")
            p.parent.mkdir(parents=True, exist_ok=True)
            _osa(
                'tell application "Pages" to save front document in '
                f'(POSIX file "{_esc(str(p))}")'
            )
            return f"saved to {p}"
        _osa('tell application "Pages" to save front document')
        return "saved"

    def close(self) -> str:
        """Close the front document (saving)."""
        _osa('tell application "Pages" to close front document saving yes')
        return "closed the document"

    # ── selection, copy / cut / paste (via the clipboard) ────────────────
    def select_all(self) -> str:
        self._activate()
        self._keystroke("a", using="command down")
        return "selected everything"

    def copy_selection(self) -> str:
        """Copy the current selection to the clipboard (⌘C)."""
        self._activate()
        self._keystroke("c", using="command down")
        return "copied the selection"

    def cut_selection(self) -> str:
        """Cut the current selection to the clipboard (⌘X)."""
        self._activate()
        self._keystroke("x", using="command down")
        return "cut the selection"

    def delete_selection(self) -> str:
        """Delete the current selection without putting it on the clipboard."""
        self._activate()
        _osa(
            'tell application "System Events" to tell process "Pages" to key code 51'
        )  # key code 51 = Delete
        return "deleted the selection"

    def paste(self) -> str:
        """Paste the clipboard into the document at the cursor (⌘V)."""
        self._activate()
        self._paste()
        return "pasted"

    # ── inserting content at the cursor ──────────────────────────────────
    def insert_text(self, text: str, *, style: str = "body") -> str:
        """Insert text AT THE CURSOR (keeps existing content). `style` one of
        title/subtitle/heading/subheading/body — non-body styles paste as large bold
        text so they read as headings."""
        self._activate()
        if style and style != "body" and style in _STYLE_SIZES:
            self._paste_rtf(text, size=_STYLE_SIZES[style])
        else:
            self.set_clipboard(text)
            self._paste()
        return f"inserted the {style if style in _STYLE_SIZES else 'body'} text"

    def add_title(self, text: str) -> str:
        """Insert a big document title at the cursor, then start a new line."""
        return self._insert_block(text, "title")

    def add_heading(self, text: str, level: int = 1) -> str:
        """Insert a heading (level 1 = larger, 2 = smaller) then a new line."""
        return self._insert_block(text, "heading" if level <= 1 else "subheading")

    def _insert_block(self, text: str, style: str) -> str:
        self.insert_text(text, style=style)
        _osa(  # Return (key code 36) so text typed next sits on a fresh, un-styled line
            'tell application "System Events" to tell process "Pages" to key code 36'
        )
        return f"added the {style}"

    def _paste_rtf(self, text: str, *, size: int, bold: bool = True) -> None:
        """Put a styled RTF run on the clipboard and paste it, so the inserted text
        carries real formatting (bold / large) rather than plain characters."""
        b_open, b_close = ("\\b ", "\\b0 ") if bold else ("", "")
        rtf = (
            "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Helvetica;}}\n"
            f"\\f0\\fs{size * 2}{b_open}{_rtf_escape(text)}{b_close}}}"
        )  # RTF font size is in half-points, hence size*2
        with tempfile.NamedTemporaryFile("w", suffix=".rtf", delete=False) as f:
            f.write(rtf)
            path = f.name
        _osa(
            f'set the clipboard to (read (POSIX file "{_esc(path)}") as «class RTF »)'
        )
        self._paste()

    # ── tables & images ──────────────────────────────────────────────────
    def insert_table(self) -> str:
        """Insert a table at the cursor via Pages' Insert ▸ Table menu."""
        self._activate()
        _osa(
            'tell application "System Events" to tell process "Pages"\n'
            '  click menu item 1 of menu 1 of menu item "Table" of menu "Insert" '
            "of menu bar 1\n"
            "end tell"
        )
        return "inserted a table"

    def insert_image(self, path: str) -> str:
        """Insert an image FILE at the cursor by copying it to the clipboard as a
        picture and pasting it in."""
        self._activate()
        try:
            _osa(
                f'set the clipboard to (read (POSIX file "{_esc(path)}") as picture)'
            )
        except PagesError as e:
            raise PagesError(f"I couldn't read that image ({path}), sir") from e
        self._paste()
        return "inserted the image"
