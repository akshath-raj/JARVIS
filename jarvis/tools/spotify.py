"""Spotify control for macOS — background & non-disruptive.

Design:
  * Playback is driven by AppleScript against the local Spotify.app, launched in
    the BACKGROUND (never brought to the foreground), so it doesn't interrupt what
    the user is doing.
  * Track/playlist SEARCH uses the Spotify Web API. Catalog search needs only the
    Client-Credentials flow (no login). PLAYLIST read/modify (play a named
    playlist, add the current song to a playlist) needs user OAuth — see
    jarvis.spotify_auth for the one-time setup.

Only a song title / playlist name ever leaves the machine.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

# Accept track / playlist / album / artist context URIs; validated before any
# interpolation into AppleScript (prevents injection via the one templated value).
_URI_RE = re.compile(r"^spotify:(track|playlist|album|artist):[A-Za-z0-9]+$")

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SEARCH_URL = "https://api.spotify.com/v1/search"
_API = "https://api.spotify.com/v1"

TOKENS_PATH = Path.home() / ".jarvis" / "spotify_tokens.json"
OAUTH_SCOPES = "playlist-read-private playlist-modify-private playlist-modify-public"


@dataclass
class Track:
    uri: str
    name: str
    artist: str

    @property
    def label(self) -> str:
        return f"{self.name} by {self.artist}"


@dataclass
class PlayResult:
    label: str
    source: str  # "app" or "web"


class SpotifyError(RuntimeError):
    pass


class SpotifyController:
    def __init__(self, client_id: str = "", client_secret: str = ""):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    @property
    def has_web_api(self) -> bool:
        return bool(self._client_id and self._client_secret)

    # ── AppleScript (local, background) ───────────────────────────────────
    @staticmethod
    def _osa(script: str) -> str:
        proc = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            raise SpotifyError(f"AppleScript error: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def ensure_running(self) -> None:
        """Start Spotify WITHOUT stealing focus.

        AppleScript `launch` starts the app in the background and does NOT bring it
        to the foreground (unlike `activate` or `open -a`, which steal focus). If
        it's already running, this is a no-op.
        """
        try:
            running = self._osa('application "Spotify" is running') == "true"
        except SpotifyError:
            running = False
        if running:
            return
        self._osa('launch application "Spotify"')
        for _ in range(20):
            try:
                if self._osa('application "Spotify" is running') == "true":
                    return
            except SpotifyError:
                pass
            time.sleep(0.25)

    def _frontmost_app(self) -> str:
        try:
            return self._osa(
                'tell application "System Events" to return name of first '
                "application process whose frontmost is true"
            )
        except SpotifyError:
            return ""

    def _restore_focus(self, prev: str) -> None:
        """Bring the previously-focused app back to the front (Spotify activates
        itself when told to play, which would otherwise steal focus)."""
        if prev and prev != "Spotify":
            try:
                self._osa(f'tell application "{prev}" to activate')
            except SpotifyError:
                pass

    def _play_context(self, uri: str) -> None:
        if not _URI_RE.match(uri):
            raise SpotifyError(f"Refusing to play malformed URI: {uri!r}")
        prev = self._frontmost_app()
        self._osa(f'tell application "Spotify" to play track "{uri}"')
        # Spotify activates itself asynchronously after 'play track' returns, so
        # wait for it to come forward, then hand focus back to the user's app.
        time.sleep(0.4)
        self._restore_focus(prev)

    def play_uri(self, uri: str) -> None:
        self._play_context(uri)

    def pause(self) -> None:
        self._osa('tell application "Spotify" to pause')

    def resume(self) -> None:
        self._osa('tell application "Spotify" to play')

    def next_track(self) -> None:
        self._osa('tell application "Spotify" to next track')

    def previous_track(self) -> None:
        self._osa('tell application "Spotify" to previous track')

    def set_volume(self, level: int) -> None:
        level = max(0, min(100, int(level)))
        self._osa(f'tell application "Spotify" to set sound volume to {level}')

    def set_repeat(self, enabled: bool) -> None:
        self._osa(f'tell application "Spotify" to set repeating to {str(enabled).lower()}')

    def loop_current(self) -> str:
        """Put the CURRENTLY playing song on repeat (isolate it, then repeat).

        Spotify's AppleScript only exposes a boolean `repeating` (repeat the
        current context), so to loop just this one song we replay it on its own so
        the context is a single track, then enable repeat.
        """
        label = self.current_track()
        uri = self.current_track_uri()
        self._play_context(uri)
        time.sleep(0.6)  # let the new context settle before enabling repeat
        self.set_repeat(True)
        return label

    def set_shuffle(self, enabled: bool) -> None:
        self._osa(f'tell application "Spotify" to set shuffling to {str(enabled).lower()}')

    def current_track(self) -> str:
        return self._osa(
            'tell application "Spotify" to return '
            '(name of current track) & " by " & (artist of current track)'
        )

    def current_track_uri(self) -> str:
        uri = self._osa('tell application "Spotify" to return id of current track')
        if not _URI_RE.match(uri):
            raise SpotifyError("No valid current track.")
        return uri

    def player_state(self) -> str:
        try:
            return self._osa('tell application "Spotify" to return player state as string')
        except SpotifyError:
            return "stopped"

    # ── Web API: app-level (client credentials) ───────────────────────────
    def _app_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token
        if not self.has_web_api:
            raise SpotifyError("No Spotify Web API credentials configured.")
        creds = f"{self._client_id}:{self._client_secret}".encode()
        resp = requests.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": "Basic " + base64.b64encode(creds).decode()},
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyError(f"Token request failed ({resp.status_code}): {resp.text}")
        p = resp.json()
        self._token = p["access_token"]
        self._token_expires_at = time.time() + p.get("expires_in", 3600)
        return self._token

    def search_track(self, query: str) -> Track | None:
        if not (query or "").strip():
            raise SpotifyError("empty search query")
        resp = requests.get(
            _SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {self._app_token()}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyError(f"Search failed ({resp.status_code}): {resp.text}")
        items = resp.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        it = items[0]
        return Track(it["uri"], it["name"], ", ".join(a["name"] for a in it["artists"]))

    # ── Web API: user-scoped (OAuth refresh token) ────────────────────────
    def _user_token(self) -> str:
        if not TOKENS_PATH.exists():
            raise SpotifyError(
                "Playlist features need a one-time login: run "
                "`python -m jarvis.spotify_auth` in your terminal."
            )
        data = json.loads(TOKENS_PATH.read_text())
        if data.get("access_token") and time.time() < data.get("expires_at", 0) - 30:
            return data["access_token"]
        # refresh
        creds = f"{self._client_id}:{self._client_secret}".encode()
        resp = requests.post(
            _TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": data["refresh_token"]},
            headers={"Authorization": "Basic " + base64.b64encode(creds).decode()},
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyError(f"Token refresh failed ({resp.status_code}): {resp.text}")
        p = resp.json()
        data["access_token"] = p["access_token"]
        data["expires_at"] = time.time() + p.get("expires_in", 3600)
        if p.get("refresh_token"):
            data["refresh_token"] = p["refresh_token"]
        TOKENS_PATH.write_text(json.dumps(data))
        return data["access_token"]

    def _me_id(self, token: str) -> str:
        r = requests.get(f"{_API}/me", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        r.raise_for_status()
        return r.json()["id"]

    def list_playlists(self) -> list[tuple[str, int | None]]:
        """Return the user's playlists as (name, track_count). track_count is None
        when Spotify won't disclose it (some tokens/apps get 403 on track reads)."""
        token = self._user_token()
        hdr = {"Authorization": f"Bearer {token}"}
        rows: list[tuple[str, str, int | None]] = []
        url = f"{_API}/me/playlists?limit=50"
        while url:
            r = requests.get(url, headers=hdr, timeout=10)
            if r.status_code != 200:
                raise SpotifyError(f"Playlist list failed ({r.status_code}): {r.text}")
            body = r.json()
            for pl in body.get("items", []):
                name = (pl.get("name") or "").strip()
                if name:
                    rows.append((pl["id"], name, (pl.get("tracks") or {}).get("total")))
            url = body.get("next")
        # /me/playlists sometimes returns tracks.total = null; try to fill it.
        out: list[tuple[str, int | None]] = []
        for pid, name, total in rows:
            if total is None:
                try:
                    rr = requests.get(f"{_API}/playlists/{pid}?fields=tracks(total)", headers=hdr, timeout=10)
                    if rr.status_code == 200:
                        total = rr.json().get("tracks", {}).get("total")
                except Exception:
                    pass
            out.append((name, total))
        return out

    def find_playlist(self, name: str) -> tuple[str, str] | None:
        """Return (uri, name) of the user's playlist best-matching `name`."""
        token = self._user_token()
        want = name.strip().lower()
        url = f"{_API}/me/playlists?limit=50"
        best = None
        while url:
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
            if r.status_code != 200:
                raise SpotifyError(f"Playlist list failed ({r.status_code}): {r.text}")
            body = r.json()
            for pl in body.get("items", []):
                pname = (pl.get("name") or "").strip()
                low = pname.lower()
                if low == want:
                    return pl["uri"], pname
                if best is None and want in low:
                    best = (pl["uri"], pname)
            url = body.get("next")
        return best

    def add_current_to_playlist(self, playlist_name: str) -> tuple[str, str]:
        """Add the currently playing track to a named playlist. Returns (track, playlist)."""
        track_uri = self.current_track_uri()
        track_label = self.current_track()
        found = self.find_playlist(playlist_name)
        if not found:
            raise SpotifyError(f"No playlist named '{playlist_name}'.")
        pl_uri, pl_name = found
        pl_id = pl_uri.split(":")[-1]
        token = self._user_token()
        r = requests.post(
            f"{_API}/playlists/{pl_id}/tracks",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"uris": [track_uri]},
            timeout=10,
        )
        if r.status_code == 403:
            raise SpotifyError(
                "Spotify denied the playlist edit (403). Your token has the right "
                "scopes, so this is an app restriction: in the Spotify developer "
                "dashboard, open your app's User Management and add your Spotify "
                "account to the allowlist (required in Development Mode)."
            )
        if r.status_code not in (200, 201):
            raise SpotifyError(f"Add to playlist failed ({r.status_code}): {r.text}")
        return track_label, pl_name

    # ── High-level ────────────────────────────────────────────────────────
    def play_query(self, query: str, mode: str = "auto", loop: bool = False) -> PlayResult:
        """Search for a song and play it in the background. `loop` sets repeat.

        The app-UI path uses System Events keystrokes (to the frontmost app), so it
        is only used when there's no Web API key — otherwise `auto` uses the fully
        background web path that never steals focus or keystrokes.
        """
        use_app = mode == "app" or (mode == "auto" and not self.has_web_api)
        if use_app:
            try:
                if self._try_play_via_app(query):
                    if loop:
                        self.set_repeat(True)
                    return PlayResult(label=self.current_track(), source="app")
            except SpotifyError:
                pass
            if mode == "app":
                raise SpotifyError("Couldn't play from the Spotify app UI.")

        if not self.has_web_api:
            raise SpotifyError(
                "The app couldn't auto-play and no Spotify API key is set for search."
            )
        track = self.search_track(query)
        if track is None:
            raise SpotifyError(f"No results for '{query}'.")
        self.ensure_running()
        self.play_uri(track.uri)
        if loop:
            time.sleep(0.6)  # let playback start before enabling repeat
            self.set_repeat(True)
        return PlayResult(label=track.label, source="web")

    def play_playlist(self, name: str, loop: bool = False) -> str:
        """Play one of the user's playlists by name (background)."""
        found = self.find_playlist(name)
        if not found:
            raise SpotifyError(f"No playlist named '{name}'.")
        uri, pl_name = found
        self.ensure_running()
        self._play_context(uri)
        if loop:
            time.sleep(0.6)
            self.set_repeat(True)
        return pl_name

    def _try_play_via_app(self, query: str) -> bool:
        """Background-friendly app search+play; verifies playback actually started."""
        before_id = ""
        try:
            before_id = self.current_track_uri()
        except SpotifyError:
            pass
        before_state = self.player_state()

        self.ensure_running()
        encoded = urllib.parse.quote(query)
        # `open -g` keeps Spotify in the background (does not steal focus).
        subprocess.run(["open", "-g", f"spotify:search:{encoded}"], timeout=10)
        time.sleep(1.6)
        try:
            self._osa(
                'tell application "System Events" to key code 48\n'  # Tab
                "delay 0.3\n"
                'tell application "System Events" to key code 36'  # Return
            )
        except SpotifyError:
            pass
        time.sleep(1.0)
        after_id = ""
        try:
            after_id = self.current_track_uri()
        except SpotifyError:
            pass
        playing = self.player_state() == "playing"
        changed = bool(after_id) and after_id != before_id
        return playing and (changed or before_state != "playing")
