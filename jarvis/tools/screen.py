"""Screen capture + visual understanding.

`ScreenController.capture()` takes a silent, full-screen screenshot with the macOS
`screencapture` utility (no shutter sound, no UI). `explain()` sends that image to
an OpenAI vision model and returns a detailed description / answer.

This is the only place JARVIS looks at your screen, and it does so on demand: a
screenshot is taken, sent to OpenAI for the single request, then the temp file is
removed. It requires the app running JARVIS (your terminal) to have macOS **Screen
Recording** permission — System Settings ▸ Privacy & Security ▸ Screen Recording.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile


class ScreenError(RuntimeError):
    pass


# The frontmost window's title usually contains the open document's name. These
# are the apps whose title = the document (Preview/Adobe for PDFs, Office/iWork
# for slides & docs, browsers for web docs). Used to know which of several open
# documents is the LIVE one on screen.
_FRONTMOST_SCRIPT = """
tell application "System Events"
  set p to first application process whose frontmost is true
  set appName to name of p
  try
    set winTitle to name of front window of p
  on error
    set winTitle to ""
  end try
end tell
return appName & "\n" & winTitle
"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (it may wrap it in prose)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class ScreenController:
    """Capture the screen and explain it with a vision model (OpenAI gpt-4o or
    Cerebras Gemma 4 — set `base_url` to the Cerebras endpoint for the latter)."""

    def __init__(self, *, vision_model: str, openai_api_key: str = "",
                 base_url: str = "") -> None:
        self._vision_model = vision_model
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url  # empty = OpenAI; else an OpenAI-compatible host

    def _vision_call(self, instruction: str, image_b64: str, *, max_output: int):
        """One OpenAI-compatible vision request. Works for OpenAI and Cerebras."""
        from openai import OpenAI

        from jarvis.openai_compat import chat_complete

        client = OpenAI(api_key=self._api_key, base_url=self._base_url or None)
        # Cerebras Gemma reasons before answering; keep it low so vision stays fast.
        reasoning = "low" if self._base_url else None
        return chat_complete(
            client,
            model=self._vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}", "detail": "high"}},
                ],
            }],
            max_output=max_output,
            reasoning_effort=reasoning,
        )

    # ── capture ──────────────────────────────────────────────────────────
    def capture(self) -> str:
        """Take a silent full-screen screenshot; return the PNG file path.

        Raises ScreenError if `screencapture` fails (e.g. Screen Recording
        permission not granted).
        """
        fd, path = tempfile.mkstemp(prefix="jarvis-screen-", suffix=".png")
        os.close(fd)
        # -x: no sound. -t png: format. Captures the main display to `path`.
        proc = subprocess.run(
            ["screencapture", "-x", "-t", "png", path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            err = proc.stderr.strip() or "screencapture produced no image"
            raise ScreenError(
                f"couldn't capture the screen ({err}); check Screen Recording permission"
            )
        return path

    # ── explain ──────────────────────────────────────────────────────────
    def explain(self, question: str = "") -> str:
        """Screenshot the screen and ask an OpenAI vision model to describe it.

        `question` is what the user wants to know; when empty, JARVIS gives a
        thorough rundown of everything visible. Returns the model's answer.
        """
        answer, _ = self.analyse(question)
        return answer

    def analyse(self, question: str = "") -> tuple[str, str]:
        """Like `explain`, but also returns the base64 PNG of the screenshot so the
        HUD can display the captured screen alongside the answer.

        Returns (answer, image_base64).
        """
        if not self._api_key:
            raise ScreenError("no OPENAI_API_KEY set — I can't analyse the screen")
        path = self.capture()
        try:
            b64 = base64.b64encode(open(path, "rb").read()).decode()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        prompt = question.strip() or (
            "Describe in detail everything shown on this screen."
        )
        instruction = (
            "You are JARVIS looking at the user's screen. Answer their question about "
            "what's displayed, clearly and in detail: name the app/website, read the "
            "relevant text, and explain what's going on. The reply is spoken aloud, so "
            "use plain prose (no markdown). Question: " + prompt
        )
        try:
            resp = self._vision_call(instruction, b64, max_output=700)
        except Exception as e:  # network / API / auth error
            raise ScreenError(f"vision model error: {e}") from e

        answer = (resp.choices[0].message.content or "").strip()
        return (answer or "I couldn't make out anything meaningful on the screen, sir.", b64)

    # ── which document is live on screen ──────────────────────────────────
    def frontmost_window(self) -> dict:
        """Return {'app', 'title'} for the currently focused window. The title
        typically holds the open document's name, so JARVIS knows which of
        several open documents the user is actually looking at."""
        try:
            out = subprocess.run(
                ["osascript", "-e", _FRONTMOST_SCRIPT],
                capture_output=True, text=True, timeout=5,
            )
            lines = (out.stdout or "").splitlines()
            return {
                "app": lines[0].strip() if lines else "",
                "title": lines[1].strip() if len(lines) > 1 else "",
            }
        except (subprocess.SubprocessError, OSError):
            return {"app": "", "title": ""}

    def read_screen_context(self, question: str = "") -> dict:
        """Screenshot the focused window and use vision to pin down WHAT the user
        is looking at: the document name, the page/slide number showing, the
        subtopic/heading, and a verbatim snippet of the visible body text. The
        snippet lets JARVIS locate this exact spot inside the indexed document so
        it can pull the surrounding pages of the same subtopic. Returns a dict:
        {document, page, subtopic, snippet, window_title, app, image_b64}."""
        if not self._api_key:
            raise ScreenError("no OPENAI_API_KEY set — I can't read the screen")
        win = self.frontmost_window()
        path = self.capture()
        try:
            b64 = base64.b64encode(open(path, "rb").read()).decode()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        instruction = (
            "You are JARVIS looking at the user's screen, focused on a document, PDF, "
            "or slide deck. Return ONLY a compact JSON object (no prose) with keys: "
            '"document" (the document title/filename if visible, else ""), '
            '"page" (the page or slide NUMBER currently shown as an integer, else null), '
            '"subtopic" (the heading/section/topic the visible content belongs to, else ""), '
            '"snippet" (a verbatim one-to-two sentence excerpt of the main visible body '
            'text — used to locate this spot in the file). '
            f"The window title is {win.get('title','')!r}. "
            + (f"The user asked: {question!r}. " if question else "")
        )
        try:
            resp = self._vision_call(instruction, b64, max_output=400)
        except Exception as e:  # network / API / auth error
            raise ScreenError(f"vision model error: {e}") from e

        data = _extract_json(resp.choices[0].message.content or "")
        page = data.get("page")
        return {
            "document": (data.get("document") or "").strip(),
            "page": int(page) if isinstance(page, (int, float)) else 0,
            "subtopic": (data.get("subtopic") or "").strip(),
            "snippet": (data.get("snippet") or "").strip(),
            "window_title": win.get("title", ""),
            "app": win.get("app", ""),
            "image_b64": b64,
        }
