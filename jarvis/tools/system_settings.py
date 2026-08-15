"""Control the Mac's own display + sound: screen brightness, system output volume,
Dark Mode appearance, Night Shift warm tone, and the desktop wallpaper.

These drive macOS itself (not an app), so they use AppleScript/`osascript` and the
system frameworks:

* Volume — AppleScript's built-in `set volume` / `get volume settings` (no perms).
* Brightness — the `brightness` CLI (`brew install brightness`) for ABSOLUTE control
  and reads; otherwise we fall back to simulating the F1/F2 brightness keys via
  System Events (needs Accessibility permission, steps only, can't read the level).
* Appearance — System Events `appearance preferences` (Dark Mode).
* Night Shift (warm tone) — the `nightlight` CLI (`brew install smudge/smudge/nightlight`)
  if present; otherwise a no-op that reports it's unavailable.
* Wallpaper — System Events desktops.

Everything is best-effort and reversible: callers can read a value, change it, and
put it back. Failures raise SettingsError with a spoken-friendly message.
"""
from __future__ import annotations

import shutil
import subprocess

# The internal keyboard's brightness keys, for the no-CLI fallback. macOS maps
# F1/F2 to these key codes regardless of the "use F-keys" setting.
_KEY_BRIGHTNESS_UP = 144
_KEY_BRIGHTNESS_DOWN = 145
# The brightness keys move in ~1/16 steps, so 16 presses spans the whole range.
_BRIGHTNESS_STEPS = 16


class SettingsError(RuntimeError):
    pass


def _osa(script: str, timeout: int = 15) -> str:
    p = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
    )
    if p.returncode != 0:
        raise SettingsError(p.stderr.strip() or "AppleScript failed")
    return p.stdout.strip()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SystemSettings:
    def __init__(self) -> None:
        # `brightness` gives absolute get/set; without it we can only step the keys.
        self._brightness_cli = shutil.which("brightness")
        self._nightlight_cli = shutil.which("nightlight")

    # ── system output volume (0–100) ─────────────────────────────────────
    def get_volume(self) -> int:
        try:
            return int(float(_osa("output volume of (get volume settings)")))
        except (SettingsError, ValueError):
            return 0

    def set_volume(self, level: int) -> int:
        level = int(_clamp(level, 0, 100))
        _osa(f"set volume output volume {level}")
        if level > 0:
            _osa("set volume without output muted")
        return level

    def nudge_volume(self, up: bool, step: int = 10) -> int:
        cur = self.get_volume()
        return self.set_volume(cur + step if up else cur - step)

    def is_muted(self) -> bool:
        try:
            return _osa("output muted of (get volume settings)").lower() == "true"
        except SettingsError:
            return False

    def set_muted(self, muted: bool) -> bool:
        _osa(f"set volume {'with' if muted else 'without'} output muted")
        return muted

    def toggle_mute(self) -> bool:
        return self.set_muted(not self.is_muted())

    # ── screen brightness (fraction 0.0–1.0) ─────────────────────────────
    @property
    def can_read_brightness(self) -> bool:
        return self._brightness_cli is not None

    def get_brightness(self) -> float | None:
        """Current brightness 0.0–1.0, or None if the `brightness` CLI is absent."""
        if not self._brightness_cli:
            return None
        try:
            out = subprocess.run(
                [self._brightness_cli, "-l"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if "brightness" in line.lower():
                    return _clamp(float(line.strip().split()[-1]), 0.0, 1.0)
        except (OSError, ValueError):
            return None
        return None

    def set_brightness(self, fraction: float) -> float:
        """Set brightness to a 0.0–1.0 fraction. Uses the `brightness` CLI when
        available (absolute); otherwise steps the brightness keys to approximate it
        (drop to minimum, then raise to the target number of steps)."""
        fraction = _clamp(fraction, 0.0, 1.0)
        if self._brightness_cli:
            subprocess.run(
                [self._brightness_cli, f"{fraction:.2f}"],
                capture_output=True, text=True, timeout=10,
            )
            return fraction
        # No CLI: approximate by pressing keys. Floor first for a known baseline.
        self._press_brightness(_KEY_BRIGHTNESS_DOWN, _BRIGHTNESS_STEPS)
        self._press_brightness(_KEY_BRIGHTNESS_UP, round(fraction * _BRIGHTNESS_STEPS))
        return fraction

    def nudge_brightness(self, up: bool, steps: int = 2) -> float | None:
        cur = self.get_brightness()
        if self._brightness_cli and cur is not None:
            return self.set_brightness(cur + (steps / _BRIGHTNESS_STEPS) * (1 if up else -1))
        self._press_brightness(_KEY_BRIGHTNESS_UP if up else _KEY_BRIGHTNESS_DOWN, steps)
        return self.get_brightness()  # None without the CLI

    def _press_brightness(self, key_code: int, times: int) -> None:
        times = max(0, int(times))
        if times == 0:
            return
        presses = "\n".join([f"key code {key_code}"] * times)
        try:
            _osa(f'tell application "System Events"\n{presses}\nend tell')
        except SettingsError as e:
            raise SettingsError(
                "I couldn't change the brightness — grant Accessibility to the app in "
                "System Settings, or `brew install brightness` for precise control, sir."
            ) from e

    # ── appearance: Dark Mode ────────────────────────────────────────────
    def is_dark_mode(self) -> bool:
        try:
            return _osa(
                'tell application "System Events" to tell appearance preferences '
                "to get dark mode"
            ).lower() == "true"
        except SettingsError:
            return False

    def set_dark_mode(self, on: bool) -> bool:
        _osa(
            'tell application "System Events" to tell appearance preferences '
            f"to set dark mode to {'true' if on else 'false'}"
        )
        return on

    # ── Night Shift (warm, low-blue-light tone) — best effort ────────────
    @property
    def can_night_shift(self) -> bool:
        return self._nightlight_cli is not None

    def set_night_shift(self, on: bool) -> bool:
        """Turn Night Shift on/off via the `nightlight` CLI. Returns whether it was
        actually applied (False if the CLI isn't installed)."""
        if not self._nightlight_cli:
            return False
        subprocess.run(
            [self._nightlight_cli, "on" if on else "off"],
            capture_output=True, text=True, timeout=10,
        )
        return True

    # ── desktop wallpaper ────────────────────────────────────────────────
    def get_wallpaper(self) -> str | None:
        try:
            path = _osa('tell application "System Events" to get picture of current desktop')
            return path or None
        except SettingsError:
            return None

    def set_wallpaper(self, path: str) -> None:
        _osa(
            'tell application "System Events" to tell every desktop '
            f'to set picture to "{path}"'
        )
