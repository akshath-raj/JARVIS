"""Reading Mode — turn the Mac into a comfortable, distraction-light reading space.

Best-practice reading ergonomics (matched to what macOS can actually do):
* **Warm, low-blue-light tone** — Night Shift on (if available); easier on the eyes,
  better for evening reading and sleep.
* **Dark, low-glare background** — Dark Mode on, so bright white chrome doesn't glare.
* **Comfortable, dimmer brightness** — matched closer to the room, not blazing.
* **Calm ambience** — soft instrumental music on Spotify in the background, at a low
  volume that sits under your reading rather than demanding attention.

Everything the mode changes is **saved first and restored on exit**, so "stop
reading mode" returns brightness, volume, appearance, Night Shift and any music to
how they were. Each individual step is best-effort: if Spotify isn't set up or a
setting can't be changed, the rest of the mode still applies.
"""
from __future__ import annotations

import logging

from jarvis.tools.spotify import SpotifyError
from jarvis.tools.system_settings import SettingsError, SystemSettings

logger = logging.getLogger("jarvis.reading")


class ReadingMode:
    def __init__(
        self,
        *,
        system: SystemSettings,
        spotify=None,
        search_mode: str = "auto",
        brightness: float = 0.45,
        volume: int = 25,
        music_query: str = "peaceful piano instrumental for reading",
        dark_mode: bool = True,
        night_shift: bool = True,
        wallpaper: str = "",
        play_music: bool = True,
    ) -> None:
        self._system = system
        self._spotify = spotify
        self._search_mode = search_mode
        self._brightness = brightness
        self._volume = volume
        self._music_query = music_query
        self._dark_mode = dark_mode
        self._night_shift = night_shift
        self._wallpaper = wallpaper
        self._play_music = play_music

        self._active = False
        self._saved: dict = {}
        self._started_music = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self, music_query: str = "") -> str:
        """Enter reading mode. `music_query` overrides the default background music."""
        if self._active:
            return "reading mode is already on, sir"

        # Snapshot what we're about to change, so stop() can restore it.
        self._saved = {
            "brightness": self._safe(self._system.get_brightness),
            "volume": self._safe(self._system.get_volume),
            "dark_mode": self._safe(self._system.is_dark_mode),
            "wallpaper": self._safe(self._system.get_wallpaper) if self._wallpaper else None,
        }

        applied: list[str] = []

        if self._dark_mode:
            self._try(lambda: self._system.set_dark_mode(True), "dark background")
            applied.append("a dark background")
        if self._night_shift and self._system.can_night_shift:
            self._try(lambda: self._system.set_night_shift(True), "warm tone")
            applied.append("a warm tone")
        self._try(lambda: self._system.set_brightness(self._brightness), "brightness")
        applied.append("softer brightness")
        if self._wallpaper:
            self._try(lambda: self._system.set_wallpaper(self._wallpaper), "wallpaper")
        self._try(lambda: self._system.set_volume(self._volume), "volume")

        if self._play_music and self._spotify is not None:
            query = music_query.strip() or self._music_query
            try:
                self._spotify.play_query(query, self._search_mode, loop=False)
                self._started_music = True
                applied.append("some soft reading music")
            except SpotifyError as e:
                logger.info("reading mode: skipping music (%s)", e)

        self._active = True
        pretty = ", ".join(applied[:-1]) + (" and " + applied[-1] if len(applied) > 1 else applied[0])
        return f"Reading mode on — {pretty}. Enjoy, sir."

    def stop(self) -> str:
        """Exit reading mode and restore the previous settings."""
        if not self._active:
            return "reading mode isn't on, sir"

        if self._night_shift and self._system.can_night_shift:
            self._try(lambda: self._system.set_night_shift(False), "warm tone")

        prev = self._saved
        if prev.get("dark_mode") is not None:
            self._try(lambda: self._system.set_dark_mode(bool(prev["dark_mode"])), "dark mode")
        if prev.get("brightness") is not None:
            self._try(lambda: self._system.set_brightness(float(prev["brightness"])), "brightness")
        if prev.get("volume") is not None:
            self._try(lambda: self._system.set_volume(int(prev["volume"])), "volume")
        if prev.get("wallpaper"):
            self._try(lambda: self._system.set_wallpaper(str(prev["wallpaper"])), "wallpaper")

        if self._started_music and self._spotify is not None:
            try:
                self._spotify.pause()
            except SpotifyError:
                pass

        self._active = False
        self._saved = {}
        self._started_music = False
        return "Reading mode off — settings restored, sir."

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except SettingsError:
            return None

    @staticmethod
    def _try(fn, what: str) -> None:
        try:
            fn()
        except SettingsError as e:
            logger.info("reading mode: couldn't set %s (%s)", what, e)
