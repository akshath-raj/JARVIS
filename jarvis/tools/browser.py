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
import time
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
# Netflix numeric title id from a title/watch URL (the same id works for both, so
# opening /watch/<id> in the logged-in browser starts playback).
_NETFLIX_ID_RE = re.compile(r"netflix\.com/(?:title|watch)/(\d+)")
# A bare domain like "espn.com" or "news.ycombinator.com" (no scheme, no spaces).
_DOMAIN_RE = re.compile(r"^[\w-]+(\.[\w-]+)+(/\S*)?$")
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class BrowserError(RuntimeError):
    pass


class BrowserController:
    # An identical open fired within this many seconds is treated as a double-fire
    # (duplicate tool call / repeated turn) and skipped, so a show never opens twice.
    _DEDUP_WINDOW = 3.0

    def __init__(self, app: str = "Google Chrome"):
        self._app = app
        self._last_url = ""
        self._last_open_at = 0.0

    # ── low-level open (foreground) ───────────────────────────────────────
    def open_url(self, url: str) -> str:
        """Open a fully-qualified http(s) URL in Chrome (brings Chrome forward).

        A repeat of the very same URL within a few seconds is dropped: a single
        "open Netflix" can reach here twice (the model firing the tool twice, or a
        duplicate turn), and without this guard that opens the site in two tabs."""
        if not re.match(r"^https?://", url, re.I):
            raise BrowserError(f"Refusing to open non-web URL: {url!r}")
        now = time.time()
        if url == self._last_url and now - self._last_open_at < self._DEDUP_WINDOW:
            return url  # duplicate open within the window — skip the second tab
        try:
            proc = subprocess.run(
                ["open", "-a", self._app, url], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise BrowserError(f"couldn't open the browser: {e}") from e
        if proc.returncode != 0:
            raise BrowserError(f"couldn't open the browser: {proc.stderr.strip()}")
        self._last_url, self._last_open_at = url, now
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

    # ── Netflix ─────────────────────────────────────────────────────────────
    @staticmethod
    def netflix_id_from_urls(urls: list[str]) -> str:
        """First Netflix numeric title id found in a list of URLs (e.g. web-search
        results), or "" if none. The id is the same for /title/ and /watch/."""
        for u in urls or []:
            m = _NETFLIX_ID_RE.search(urllib.parse.unquote(u or ""))
            if m:
                return m.group(1)
        return ""

    def play_netflix(self, query: str, *, title_id: str = "") -> str:
        """Play a specific show/movie on Netflix in the real (logged-in) browser.

        `title_id` is the resolved Netflix numeric id (the caller resolves it via
        web search — no Netflix login/API needed). Given one, this opens the
        /watch/<id> URL, which auto-plays for a logged-in user (a series starts the
        right episode). Without a confident id it falls back to opening the Netflix
        search page so the user can pick — it never guesses a wrong title."""
        if not (query or "").strip():
            raise BrowserError("what should I play on Netflix?")
        if title_id and title_id.isdigit():
            self.open_url(f"https://www.netflix.com/watch/{title_id}")
            return f"playing '{query}' on Netflix"
        self.open_url("https://www.netflix.com/search?q=" + urllib.parse.quote(query))
        return f"I couldn't pin the exact title, sir, so I've opened the Netflix search for '{query}'."

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
