"""Spotify control for macOS.

Playback is always LOCAL (AppleScript against Spotify.app). The only tricky part
is turning a spoken phrase into a specific track, because AppleScript can PLAY a
track URI but cannot SEARCH the catalog. Two ways to bridge that gap:

  * "app"  -> drive the Spotify desktop UI: open ``spotify:search:<q>`` and try to
              play the top result. No API key, fully local, but best-effort (depends
              on Spotify's UI, which changes).
  * "web"  -> Spotify Web API search (Client-Credentials, no user login). Sends only
              a song title over the network; reliable and exact.

Default mode is "auto": try the app first (per user preference), verify that audio
actually started, and fall back to the Web API only if it didn't.
"""
from __future__ import annotations

import base64
import subprocess
import time
import urllib.parse
from dataclasses import dataclass

import requests

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SEARCH_URL = "https://api.spotify.com/v1/search"


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
    label: str      # "Bohemian Rhapsody by Queen"
    source: str     # "app" or "web"


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

    # ---- AppleScript helpers (local) -------------------------------------
    @staticmethod
    def _osa(script: str) -> str:
        proc = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            raise SpotifyError(f"AppleScript error: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def ensure_running(self) -> None:
        self._osa('tell application "Spotify" to activate')

    def player_state(self) -> str:
        # "playing" | "paused" | "stopped"
        try:
            return self._osa('tell application "Spotify" to return player state as string')
        except SpotifyError:
            return "stopped"

    def _current_id(self) -> str:
        try:
            return self._osa('tell application "Spotify" to return id of current track')
        except SpotifyError:
            return ""

    def play_uri(self, uri: str) -> None:
        self._osa(f'tell application "Spotify" to play track "{uri}"')

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

    def current_track(self) -> str:
        return self._osa(
            'tell application "Spotify" to return '
            '(name of current track) & " by " & (artist of current track)'
        )

    # ---- Web API (search) ------------------------------------------------
    def _access_token(self) -> str:
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
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + payload.get("expires_in", 3600)
        return self._token

    def search_track(self, query: str) -> Track | None:
        resp = requests.get(
            _SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {self._access_token()}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise SpotifyError(f"Search failed ({resp.status_code}): {resp.text}")
        items = resp.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        return Track(
            uri=item["uri"],
            name=item["name"],
            artist=", ".join(a["name"] for a in item["artists"]),
        )

    # ---- App-UI search+play (no key, best-effort) ------------------------
    def _try_play_via_app(self, query: str) -> bool:
        """Open in-app search and try to play the top result.

        Returns True only if playback verifiably started (a track is playing and
        it changed from before), so the caller can trust it or fall back.
        """
        before_id = self._current_id()
        before_state = self.player_state()

        self.ensure_running()
        encoded = urllib.parse.quote(query)
        subprocess.run(["open", f"spotify:search:{encoded}"], timeout=10)
        time.sleep(1.6)  # let results render

        # Best-effort: focus results and activate the top item via keyboard.
        # (Spotify is an Electron app; this may be a no-op on some versions,
        # which is why we verify playback below and fall back if needed.)
        keystroke = (
            'tell application "Spotify" to activate\n'
            'delay 0.3\n'
            'tell application "System Events" to key code 48\n'   # Tab -> into results
            'delay 0.3\n'
            'tell application "System Events" to key code 36\n'   # Return -> play
        )
        try:
            self._osa(keystroke)
        except SpotifyError:
            pass  # Accessibility not granted / keystroke failed -> verify anyway

        time.sleep(1.0)
        after_id = self._current_id()
        playing = self.player_state() == "playing"
        changed = after_id and after_id != before_id
        started = before_state != "playing"
        return playing and (changed or started)

    # ---- High-level: hybrid resolve + play -------------------------------
    def play_query(self, query: str, mode: str = "auto") -> PlayResult:
        """Resolve a spoken query to a track and play it in the desktop app.

        mode: "app" (UI only), "web" (Web API only), or "auto" (app first,
        Web API fallback).
        """
        if mode in ("app", "auto"):
            try:
                if self._try_play_via_app(query):
                    return PlayResult(label=self.current_track(), source="app")
            except SpotifyError:
                pass
            if mode == "app":
                raise SpotifyError(
                    "Couldn't auto-play from the Spotify app. Grant Accessibility "
                    "permission, or enable Web API fallback with a Spotify key."
                )

        # Web API fallback / explicit web mode
        if not self.has_web_api:
            raise SpotifyError(
                "The app couldn't auto-play and no Spotify API key is set for "
                "fallback. Add SPOTIFY_CLIENT_ID/SECRET, or grant Accessibility "
                "permission to Terminal."
            )
        track = self.search_track(query)
        if track is None:
            raise SpotifyError(f"No results for '{query}'.")
        self.ensure_running()
        self.play_uri(track.uri)
        return PlayResult(label=track.label, source="web")
