"""Vet downloaded images with vision + reasoning before using them in a document.

The user's rule: for every candidate image, JARVIS should (1) actually LOOK at it with
its vision model (Gemma on Cerebras, or GPT-4o on OpenAI) to understand what it shows,
then (2) have its normal reasoning model decide whether that image genuinely fits what
was asked — and if not, try another candidate. This stops "deepfake" from pulling an
unrelated Venn diagram just because the filename matched.

`ImageVetter.describe()` is the vision step; `judge()` is the reasoning step; and
`accept_for(intent)` returns a callback that `SafeImageFetcher.fetch_one` calls per
candidate, so vetting slots straight into the existing safe-download loop. Both model
calls are best-effort: on any error we fail OPEN (accept the image) so a flaky model
never blocks the document — safety (which host, which bytes) is already guaranteed by
the fetcher; vetting only improves relevance.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("jarvis.image_vetting")


def _mime(data: bytes) -> str:
    """Sniff the image MIME from magic bytes so the data URL is labelled correctly
    (a JPEG sent as image/png makes the vision API reject it)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class ImageVetter:
    """Describe an image (vision) and judge its fit for an intent (reasoning)."""

    def __init__(self, *, vision_model: str, text_model: str, api_key: str = "",
                 base_url: str = "") -> None:
        self.vision_model = vision_model
        self.text_model = text_model
        self._api_key = api_key
        self._base_url = base_url  # "" = OpenAI; else an OpenAI-compatible host (Cerebras)

    @classmethod
    def from_config(cls) -> "ImageVetter":
        """Build from JARVIS config: Cerebras (Gemma vision + gpt-oss text) when keyed,
        else OpenAI (gpt-4o vision + the cloud agent model)."""
        from jarvis.config import config
        if config.agent_provider == "cerebras" and config.cerebras_api_key:
            return cls(vision_model=config.cerebras_vision_model,
                       text_model=config.cerebras_model,
                       api_key=config.cerebras_api_key, base_url=config.cerebras_base_url)
        return cls(vision_model=config.vision_model, text_model=config.cloud_agent_model,
                   api_key=config.openai_api_key)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key, base_url=self._base_url or None)

    # ── vision: what IS this image? ──────────────────────────────────────
    def describe(self, path: str) -> str:
        """Return a short factual description of the image (subject, kind, any text)."""
        from jarvis.openai_compat import chat_complete
        data = Path(path).expanduser().read_bytes()
        b64 = base64.b64encode(data).decode()
        resp = chat_complete(
            self._client(),
            model=self.vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text":
                    "Describe this image factually in one or two sentences: the main "
                    "subject, whether it is a photo / diagram / chart / illustration / "
                    "screenshot, and any prominent text. Be concise."},
                {"type": "image_url", "image_url": {
                    "url": f"data:{_mime(data)};base64,{b64}", "detail": "low"}},
            ]}],
            max_output=400,
            reasoning_effort="low" if self._base_url else None,
        )
        return (resp.choices[0].message.content or "").strip()

    # ── reasoning: does it FIT what was asked? ───────────────────────────
    def judge(self, intent: str, title: str, description: str) -> tuple[bool, str]:
        """Decide whether the described image fits `intent`. Returns (fit, reason)."""
        from jarvis.openai_compat import chat_complete
        resp = chat_complete(
            self._client(),
            model=self.text_model,
            messages=[{"role": "user", "content":
                "You are choosing an image for a document. The user wants an image of: "
                f"\"{intent}\".\nA candidate image is titled \"{title}\" and looks like: "
                f"\"{description}\".\nWould this image be a good, relevant fit for what "
                "the user asked? Reply ONLY with JSON: "
                '{"fit": true|false, "reason": "<short>"}.'}],
            max_output=200,
            temperature=0,
            reasoning_effort="low" if self._base_url else None,
        )
        data = _extract_json(resp.choices[0].message.content or "")
        fit = bool(data.get("fit", True))
        return fit, str(data.get("reason", "")).strip()

    # ── the fetch_one callback ───────────────────────────────────────────
    def accept_for(self, intent: str):
        """Return an `accept(path, result) -> bool` for SafeImageFetcher.fetch_one:
        look at the image, judge fit, accept only if it fits. Fails OPEN on error."""
        def accept(path: str, result) -> bool:
            try:
                desc = self.describe(path)
                fit, reason = self.judge(intent, getattr(result, "title", ""), desc)
                logger.info("vet %s → fit=%s (%s)", getattr(result, "title", path), fit, reason)
                return fit
            except Exception as e:  # model/network hiccup → don't block the document
                logger.warning("image vetting failed, accepting image: %s", e)
                return True
        return accept
