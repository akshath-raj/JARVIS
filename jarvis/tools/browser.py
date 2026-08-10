"""Chrome control for macOS — open sites and play videos in the browser.

Design (mirrors the Spotify controller's split, but the opposite focus policy):
  * Opening/navigating is driven by `open -a "Google Chrome" <url>`, which brings
    Chrome to the FOREGROUND on purpose — the user asked to open/watch something,
    so unlike Spotify (kept in the background) we want the browser in front.
  * To PLAY a specific YouTube video we resolve the top result's video id by
    fetching the public results page and reading the first `videoId` — no API key
    or login. Only the search query / URL leaves the machine (same privacy posture
    as the Spotify catalog search); the user's voice never goes out.

Kept entirely separate from the music pipeline: no shared state, its own module.
"""
from __future__ import annotations

import re
import subprocess
import urllib.parse

import requests

# Friendly names → URLs. Unknown names fall back to a Google search.
SITE_ALIASES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "youtube shorts": "https://www.youtube.com/shorts",
    "shorts": "https://www.youtube.com/shorts",
    "instagram": "https://www.instagram.com",
    "insta": "https://www.instagram.com",
    "ig": "https://www.instagram.com",
    "instagram reels": "https://www.instagram.com/reels/",
    "reels": "https://www.instagram.com/reels/",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "gmail": "https://mail.google.com",
    "email": "https://mail.google.com",
    "google": "https://www.google.com",
    "maps": "https://www.google.com/maps",
    "google maps": "https://www.google.com/maps",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "calendar": "https://calendar.google.com",
    "github": "https://github.com",
    "netflix": "https://www.netflix.com",
    "spotify": "https://open.spotify.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "whatsapp": "https://web.whatsapp.com",
    "amazon": "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
}

# 11-char YouTube video id as embedded in page JSON: "videoId":"XXXXXXXXXXX".
_VIDEO_ID_RE = re.compile(r'"videoId":"([\w-]{11})"')
# A bare domain like "espn.com" or "news.ycombinator.com" (no scheme, no spaces).
_DOMAIN_RE = re.compile(r"^[\w-]+(\.[\w-]+)+(/\S*)?$")
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class BrowserError(RuntimeError):
    pass


class BrowserController:
    def __init__(self, app: str = "Google Chrome"):
        self._app = app

    # ── low-level open (foreground) ───────────────────────────────────────
    def open_url(self, url: str) -> str:
        """Open a fully-qualified http(s) URL in Chrome (brings Chrome forward)."""
        if not re.match(r"^https?://", url, re.I):
            raise BrowserError(f"Refusing to open non-web URL: {url!r}")
        try:
            proc = subprocess.run(
                ["open", "-a", self._app, url], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise BrowserError(f"couldn't open the browser: {e}") from e
        if proc.returncode != 0:
            raise BrowserError(f"couldn't open the browser: {proc.stderr.strip()}")
        return url

    # ── site resolution ───────────────────────────────────────────────────
    @staticmethod
    def resolve_site(name: str) -> str:
        """Turn a spoken name into a URL: alias → bare domain → Google search."""
        q = (name or "").strip()
        low = q.lower().strip(" .!?")
        if not low:
            return "https://www.google.com"
        if low in SITE_ALIASES:
            return SITE_ALIASES[low]
        if re.match(r"^https?://", q, re.I):
            return q
        if _DOMAIN_RE.match(low):
            return "https://" + low
        return "https://www.google.com/search?q=" + urllib.parse.quote(q)

    def open_site(self, name: str) -> str:
        """Open a named site (youtube, instagram, github, …) or search for it."""
        return self.open_url(self.resolve_site(name))

    # ── YouTube ───────────────────────────────────────────────────────────
    def _get(self, url: str) -> str:
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, timeout=15)
        except requests.RequestException as e:
            raise BrowserError(f"couldn't reach {url}: {e}") from e
        if r.status_code != 200:
            raise BrowserError(f"request to {url} failed ({r.status_code})")
        return r.text

    def _first_video_id(self, page_url: str) -> str:
        m = _VIDEO_ID_RE.search(self._get(page_url))
        if not m:
            raise BrowserError("couldn't find a video on that page")
        return m.group(1)

    def play_youtube(self, query: str) -> str:
        """Search YouTube for `query` and open the top result (it autoplays)."""
        if not (query or "").strip():
            raise BrowserError("what should I play on YouTube?")
        results = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        vid = self._first_video_id(results)
        self.open_url(f"https://www.youtube.com/watch?v={vid}")
        return f"playing '{query}' on YouTube"

    @staticmethod
    def _channel_handle(channel: str) -> str:
        # "Mr Beast" / "@MrBeast" -> "MrBeast"
        return re.sub(r"[^\w]", "", channel.lstrip("@"))

    def latest_channel_video(self, channel: str) -> str:
        """Open the newest upload from a YouTube channel.

        Note: without a YouTube login/API we can't read your personal watch
        history, so "a video I haven't watched" resolves to the channel's LATEST
        upload — nearly always the freshest/unwatched one.
        """
        if not (channel or "").strip():
            raise BrowserError("which channel?")
        handle = self._channel_handle(channel)
        vids_url = f"https://www.youtube.com/@{handle}/videos"
        try:
            vid = self._first_video_id(vids_url)
        except BrowserError:
            # Fall back to searching the channel by name if the @handle guess misses.
            vid = self._first_video_id(
                "https://www.youtube.com/results?search_query="
                + urllib.parse.quote(channel)
            )
        self.open_url(f"https://www.youtube.com/watch?v={vid}")
        return f"opening the latest {channel} video"

    # ── short-form video feeds ────────────────────────────────────────────
    def open_shorts(self) -> str:
        """Open the YouTube Shorts feed."""
        self.open_url("https://www.youtube.com/shorts")
        return "opening YouTube Shorts"

    def open_reels(self, platform: str = "instagram") -> str:
        """Open a short-form 'reels' feed. platform: instagram (default) or youtube."""
        p = (platform or "instagram").lower()
        if "you" in p or "short" in p:
            return self.open_shorts()
        self.open_url("https://www.instagram.com/reels/")
        return "opening Instagram Reels"
