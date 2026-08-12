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
import os
import subprocess
import tempfile


class ScreenError(RuntimeError):
    pass


class ScreenController:
    """Capture the screen and, optionally, explain it with an OpenAI vision model."""

    def __init__(self, *, vision_model: str, openai_api_key: str = "") -> None:
        self._vision_model = vision_model
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

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
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)
            resp = client.chat.completions.create(
                model=self._vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=700,
            )
        except Exception as e:  # network / API / auth error
            raise ScreenError(f"vision model error: {e}") from e

        answer = (resp.choices[0].message.content or "").strip()
        return (answer or "I couldn't make out anything meaningful on the screen, sir.", b64)
