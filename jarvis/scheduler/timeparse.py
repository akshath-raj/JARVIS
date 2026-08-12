"""Turn a natural-language time ("at 9pm today", "in another hour", an ISO string
the model computed) into a concrete datetime.

Order: exact ISO 8601 first (the frontier model is given the current time and can
pass an absolute datetime), then `dateparser` on the raw phrase, then a light
normalisation pass for phrasings dateparser trips on ("another" → "1").
"""
from __future__ import annotations

import datetime
import re

import dateparser


def parse_when(text: str, base: datetime.datetime | None = None) -> datetime.datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    base = base or datetime.datetime.now()

    # 1) exact ISO 8601 (e.g. the model already resolved "in an hour")
    try:
        dt = datetime.datetime.fromisoformat(text)
        return dt.replace(tzinfo=None)
    except ValueError:
        pass

    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base}
    dt = dateparser.parse(text, settings=settings)
    if dt is None:
        # "in another hour" / "another 30 minutes" → treat "another" as "1"/nothing
        norm = re.sub(r"\banother\b", "1", text, flags=re.I)
        norm = re.sub(r"\bin\s+1\s+(hour|minute|min|day)", r"in 1 \1", norm, flags=re.I)
        dt = dateparser.parse(norm, settings=settings)
    if dt is not None:
        dt = dt.replace(tzinfo=None, microsecond=0)
    return dt
