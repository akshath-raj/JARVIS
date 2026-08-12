"""AlarmPlayer — rings a macOS system sound on a loop until stopped.

`afplay` plays a file once, so a background thread replays it until `stop()` is
called (voice: "stop the alarm" / HUD: the STOP button). Thread-safe.
"""
from __future__ import annotations

import logging
import subprocess
import threading

logger = logging.getLogger("jarvis.alarm")

DEFAULT_SOUND = "/System/Library/Sounds/Sosumi.aiff"


class AlarmPlayer:
    def __init__(self, sound: str = DEFAULT_SOUND) -> None:
        self._sound = sound
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    @property
    def ringing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.ringing:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="jarvis-alarm")
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    ["afplay", "-v", "2", self._sound],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._proc.wait()
            except Exception as e:
                logger.warning("alarm playback failed: %s", e)
                self._stop.wait(1.0)
            # brief gap between rings; bail immediately if stopped
            self._stop.wait(0.4)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            self._thread = None
