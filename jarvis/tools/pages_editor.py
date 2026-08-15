"""Edit the CURRENTLY OPEN Pages document — targeted formatting changes by voice.

The user is working in Pages and says "make the heading 32pt" or "reformat this".
JARVIS operates on Pages' *front document* (the one they're looking at) and changes
formatting through AppleScript.

SAFETY — Pages' AppleScript text model is quirky: writing an *unsupported* attribute
onto a text range silently REPLACES the text with the value (e.g. `set bold … to true`
turns the paragraph into the word "true"). Verified-safe, non-destructive attributes
are only ``size`` (points), ``font`` (family name), and ``color`` ({r,g,b}, 16-bit).
We therefore whitelist exactly those; "bold"/"italic" are done by switching to a bold/
italic FONT VARIANT ("Helvetica Neue Bold"), never a bold property. Reading attributes
back isn't supported by Pages, so we read the plain paragraph text (to identify which
paragraph is the heading) and only WRITE attributes.
"""
from __future__ import annotations

import subprocess

_PARA_DELIM = "|~|"          # joins paragraphs when we read them back
_DEFAULT_FONT = "Helvetica Neue"

# spoken colour names → Pages RGB (16-bit components, 0–65535)
_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0), "white": (65535, 65535, 65535), "red": (60000, 0, 0),
    "green": (0, 40000, 0), "blue": (0, 0, 60000), "dark blue": (10000, 20000, 45000),
    "navy": (5000, 10000, 35000), "gray": (32000, 32000, 32000),
    "grey": (32000, 32000, 32000), "orange": (65535, 40000, 0),
    "purple": (35000, 0, 45000), "teal": (0, 40000, 45000),
}


class PagesEditError(RuntimeError):
    pass


def _osa(script: str, timeout: int = 20) -> str:
    p = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
    )
    if p.returncode != 0:
        err = p.stderr.strip()
        if "-1743" in err or "not allowed" in err.lower():
            raise PagesEditError(
                "I need permission to control Pages, sir — enable it under Privacy & "
                "Security ▸ Automation in System Settings."
            )
        if "-1728" in err or "Invalid index" in err or "doesn't understand" in err:
            raise PagesEditError("there's no open Pages document to edit, sir")
        raise PagesEditError(err or "Pages edit failed")
    return p.stdout.strip()


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _rgb(color: str) -> tuple[int, int, int] | None:
    """Resolve a colour name or #hex to a Pages 16-bit RGB triple (or None)."""
    c = (color or "").strip().lower()
    if not c:
        return None
    if c in _COLORS:
        return _COLORS[c]
    if c.startswith("#") and len(c) in (4, 7):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        try:
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            return (r * 257, g * 257, b * 257)  # 8-bit → 16-bit
        except ValueError:
            return None
    return None


class PagesEditor:
    """Apply formatting to the front Pages document via safe AppleScript attributes."""

    def active_document(self) -> str:
        """Name of the front (active) Pages document, or '' if none is open."""
        try:
            return _osa('tell application "Pages" to get name of front document')
        except PagesEditError:
            return ""

    def read_paragraphs(self) -> list[str]:
        """Return the front document's paragraphs as a list (index-aligned with what
        the format methods target), so the caller can tell which one is the heading."""
        out = _osa(
            'tell application "Pages"\n'
            "  set d to front document\n"
            f'  set AppleScript\'s text item delimiters to "{_PARA_DELIM}"\n'
            "  set t to (paragraphs of body text of d) as text\n"
            '  set AppleScript\'s text item delimiters to ""\n'
            "  return t\n"
            "end tell"
        )
        return out.split(_PARA_DELIM) if out else []

    # ── low-level: write only whitelisted attributes to a text range ─────
    def _apply(self, target: str, *, size=None, font=None, color=None) -> None:
        stmts = []
        if size is not None:
            stmts.append(f"set size of {target} to {int(size)}")
        if font:
            stmts.append(f'set font of {target} to "{_esc(font)}"')
        if color is not None:
            r, g, b = color
            stmts.append(f"set color of {target} to {{{r}, {g}, {b}}}")
        if not stmts:
            return
        body = "\n".join(stmts)
        _osa(f'tell application "Pages"\n{body}\nend tell')

    @staticmethod
    def _weighted(font: str | None, bold: bool, italic: bool) -> str | None:
        """Turn a bold/italic request into a font-variant name (Pages has no bold
        property). Uses the given family or the default."""
        if not (bold or italic):
            return font
        family = font or _DEFAULT_FONT
        suffix = " ".join(w for w, on in ((("Bold"), bold), (("Italic"), italic)) if on)
        return f"{family} {suffix}".strip()

    # ── public formatting operations ─────────────────────────────────────
    def format_paragraph(self, index: int, *, size=None, font=None, bold=False,
                         italic=False, color=None) -> str:
        """Format ONE paragraph (1-based). size in points; font a family name; bold/
        italic switch to that variant; color a name or #hex."""
        n = len(self.read_paragraphs())
        if index < 1 or index > n:
            raise PagesEditError(f"there's no paragraph {index} (the document has {n})")
        self._apply(
            f"paragraph {int(index)} of body text of front document",
            size=size, font=self._weighted(font, bold, italic), color=_rgb(color or ""),
        )
        bits = [b for b in (
            f"{int(size)}pt" if size else "", "bold" if bold else "",
            "italic" if italic else "", font or "", color or "",
        ) if b]
        return f"formatted paragraph {index}" + (f" ({', '.join(bits)})" if bits else "")

    def format_all(self, *, size=None, font=None, bold=False, italic=False,
                   color=None) -> str:
        """Apply formatting to the WHOLE document body."""
        self._apply(
            "body text of front document",
            size=size, font=self._weighted(font, bold, italic), color=_rgb(color or ""),
        )
        return "formatted the whole document"

    def reformat(self, *, font: str = _DEFAULT_FONT, title_size: int = 26,
                 heading_size: int = 17, body_size: int = 12) -> str:
        """Tidy the document into a consistent look. Heuristic: the first non-empty
        paragraph is the TITLE; short lines with no ending punctuation are HEADINGS;
        everything else is BODY. Titles/headings get the bold variant + accent colour,
        body gets the regular family. Best-effort — the caller can instead read the
        paragraphs and format specific ones for precise control."""
        paras = self.read_paragraphs()
        if not paras:
            raise PagesEditError("there's no open Pages document to reformat, sir")
        accent = _COLORS["dark blue"]
        seen_title = False
        counts = {"title": 0, "heading": 0, "body": 0}
        for i, text in enumerate(paras, start=1):
            t = text.strip()
            if not t:
                continue
            words = t.split()
            is_heading = len(words) <= 8 and t[-1:] not in ".!?:;," and len(t) <= 64
            if not seen_title:
                kind, seen_title = "title", True
            elif is_heading:
                kind = "heading"
            else:
                kind = "body"
            counts[kind] += 1
            if kind == "title":
                self._apply(f"paragraph {i} of body text of front document",
                            size=title_size, font=f"{font} Bold", color=accent)
            elif kind == "heading":
                self._apply(f"paragraph {i} of body text of front document",
                            size=heading_size, font=f"{font} Bold", color=accent)
            else:
                self._apply(f"paragraph {i} of body text of front document",
                            size=body_size, font=font, color=_COLORS["black"])
        return (f"reformatted the document — {counts['title']} title, "
                f"{counts['heading']} heading(s), {counts['body']} body paragraph(s)")
