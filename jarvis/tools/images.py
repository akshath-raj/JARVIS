"""Fetch images from the web SAFELY, for dropping into documents (Pages, etc.).

This is deliberately locked down. Pulling arbitrary URLs off the open web is how you
end up saving tracking pixels, HTML pages disguised as ``.jpg``, SVGs carrying
scripts, or getting bounced through a redirect chain to a hostile host. So every
download passes through hard guardrails, in order:

  1. **Allowlisted hosts only.** The URL host (and the host of EVERY redirect hop,
     and the final URL) must be in ``allowed_hosts`` — reputable, freely-licensed
     image sources (Wikimedia Commons / Wikipedia by default). A redirect that
     leaves the allowlist is rejected: no bouncing off to random websites.
  2. **HTTPS only.** No plaintext, no ``file://``/``data:`` tricks.
  3. **Declared image type.** The ``Content-Type`` must be a raster ``image/*``;
     ``image/svg+xml`` is refused (SVG is XML and can carry JavaScript).
  4. **Size cap.** The body is streamed and aborted the moment it exceeds
     ``max_bytes`` — no unbounded downloads.
  5. **Real image bytes.** The magic bytes must actually be PNG/JPEG/GIF/WebP. A
     mislabelled HTML/script payload never reaches disk.
  6. **Sandboxed destination.** Files are written only inside ``download_dir``.

Discovery uses the MediaWiki (Wikimedia Commons) search API — freely-licensed media
with a real API and no key — and every candidate URL it returns is re-checked
against the same allowlist before it's ever fetched.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

# Magic-byte signatures for the raster formats we accept → canonical extension.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)
_USER_AGENT = "JARVIS/1.0 (personal voice assistant; safe image fetch)"
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


class ImageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageResult:
    title: str          # human-readable file title (for attribution)
    url: str            # direct, allowlisted image URL to download
    source_page: str    # the description page (credit / licence)
    width: int = 0
    height: int = 0


def _sniff(head: bytes) -> str | None:
    """Return the canonical extension if `head` starts with a known image signature.
    WebP is RIFF....WEBP (checked across the first 12 bytes)."""
    for sig, ext in _MAGIC:
        if head.startswith(sig):
            return ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    return None


def _slug(text: str, limit: int = 60) -> str:
    # drop a trailing image extension first so we don't end up with foo.png.png
    if Path(text).suffix.lower() in _IMAGE_EXTS:
        text = Path(text).stem
    s = re.sub(r"[^\w.-]+", "_", text).strip("_")
    return (s[:limit] or "image").lower()


class SafeImageFetcher:
    """Search for and download web images under strict safety guardrails."""

    def __init__(
        self,
        *,
        download_dir: str,
        allowed_hosts: tuple[str, ...],
        max_bytes: int = 15 * 1024 * 1024,
        timeout: int = 20,
    ) -> None:
        self.download_dir = Path(download_dir).expanduser()
        self.allowed_hosts = tuple(h.lower() for h in allowed_hosts)
        self.max_bytes = max_bytes
        self.timeout = timeout

    # ── allowlist enforcement ────────────────────────────────────────────
    def _host_ok(self, url: str) -> bool:
        """True only for https URLs whose host is (a subdomain of) an allowed host."""
        try:
            p = urlparse(url)
        except ValueError:
            return False
        if p.scheme != "https" or not p.hostname:
            return False
        host = p.hostname.lower()
        return any(host == h or host.endswith("." + h) for h in self.allowed_hosts)

    # ── discovery (Wikimedia Commons) ────────────────────────────────────
    def search(self, query: str, count: int = 6) -> list[ImageResult]:
        """Find freely-licensed images matching `query`. Only results whose image URL
        passes the allowlist are returned (so nothing unsafe is ever surfaced)."""
        if not (query or "").strip():
            raise ImageError("empty image query")
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrnamespace": "6",  # 6 = File namespace
            "gsrlimit": str(max(1, min(count * 3, 30))),
            "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "1024",  # ask for a ≤1024px rendering, not the raw original
        }
        try:
            r = requests.get(
                _COMMONS_API, params=params,
                headers={"User-Agent": _USER_AGENT}, timeout=self.timeout,
            )
            r.raise_for_status()
            pages = (r.json().get("query", {}) or {}).get("pages", {}) or {}
        except (requests.RequestException, ValueError) as e:
            raise ImageError(f"image search failed: {e}") from e

        results: list[ImageResult] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = (info.get("mime") or "").lower()
            if not mime.startswith("image/") or mime == "image/svg+xml":
                continue
            # prefer the scaled thumbnail; fall back to the original file URL
            url = info.get("thumburl") or info.get("url") or ""
            if not self._host_ok(url):
                continue
            results.append(ImageResult(
                title=re.sub(r"^File:", "", page.get("title", "")).strip(),
                url=url,
                source_page=info.get("descriptionurl", ""),
                width=int(info.get("thumbwidth") or info.get("width") or 0),
                height=int(info.get("thumbheight") or info.get("height") or 0),
            ))
            if len(results) >= count:
                break
        return results

    def _get_with_retry(self, url: str, *, stream: bool):
        """GET with a polite retry on 429/503 (Wikimedia rate-limits bursts). Honours
        Retry-After when present, capped so we never hang the assistant."""
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    url, stream=stream, timeout=self.timeout,
                    headers={"User-Agent": _USER_AGENT},
                )
                if resp.status_code in (429, 503) and attempt < 2:
                    wait = min(float(resp.headers.get("Retry-After", "") or 1.5) + attempt, 5.0)
                    time.sleep(wait)
                    last = ImageError(f"rate-limited ({resp.status_code})")
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise ImageError(f"download failed: {last}")

    # ── guardrailed download ─────────────────────────────────────────────
    def download(self, url: str, *, name_hint: str = "") -> str:
        """Download one image URL under every guardrail. Returns the saved path.
        Raises ImageError if any check fails (host, scheme, type, size, bytes)."""
        if not self._host_ok(url):
            raise ImageError(
                f"refused: {urlparse(url).hostname or url!r} is not an allowed image "
                "source (only trusted hosts are permitted)"
            )
        resp = self._get_with_retry(url, stream=True)

        with resp:
            # a redirect must NEVER leave the allowlist — check every hop + the final URL
            for hop in list(resp.history) + [resp]:
                if not self._host_ok(hop.url):
                    raise ImageError(
                        f"refused: redirected to an untrusted host ({urlparse(hop.url).hostname})"
                    )
            ctype = (resp.headers.get("Content-Type", "").split(";")[0].strip().lower())
            if not ctype.startswith("image/") or ctype == "image/svg+xml":
                raise ImageError(f"refused: not a raster image (Content-Type {ctype or 'unknown'})")

            # stream with a hard byte cap
            chunks, total = [], 0
            for chunk in resp.iter_content(8192):
                total += len(chunk)
                if total > self.max_bytes:
                    raise ImageError(
                        f"refused: image exceeds the {self.max_bytes // (1024*1024)}MB limit"
                    )
                chunks.append(chunk)
        data = b"".join(chunks)

        ext = _sniff(data[:16])
        if ext is None:
            raise ImageError("refused: the downloaded bytes are not a valid image")

        self.download_dir.mkdir(parents=True, exist_ok=True)
        stem = _slug(name_hint or Path(urlparse(url).path).stem or "image")
        dest = self.download_dir / f"{stem}{ext}"
        i = 1
        while dest.exists():  # never overwrite
            dest = self.download_dir / f"{stem}_{i}{ext}"
            i += 1
        dest.write_bytes(data)
        return str(dest)

    # ── convenience: search → download the first safe (and accepted) match ─
    def fetch_one(self, query: str, *, accept=None) -> tuple[str, ImageResult]:
        """Find and download the best safe image for `query`. Returns (path, result).
        Tries candidates in order so a single bad file doesn't fail the whole call.

        `accept(path, result) -> bool` is an optional relevance gate (e.g. the vision+
        reasoning vetter): a candidate that downloads safely but is rejected is skipped
        and the next one tried, so JARVIS keeps looking until an image actually fits."""
        results = self.search(query, count=self.max_search_candidates)
        if not results:
            raise ImageError(f"no safe image found for “{query}”")
        last: Exception | None = None
        first_ok: tuple[str, ImageResult] | None = None
        for res in results:
            try:
                path = self.download(res.url, name_hint=res.title or query)
            except ImageError as e:
                last = e
                continue
            if accept is None or accept(path, res):
                return path, res
            if first_ok is None:
                first_ok = (path, res)  # remember a safe-but-unvetted fallback
        if first_ok is not None:
            return first_ok  # nothing was judged a perfect fit → use the best safe one
        raise ImageError(f"couldn't safely download any image for “{query}” ({last})")

    max_search_candidates = 6
