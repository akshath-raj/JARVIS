"""Control the video playing in the user's LIVE Chrome tab, and close tabs.

Unlike the headless browser agent, this drives the user's actual visible Chrome via
AppleScript, running JavaScript on the active tab's HTML5 <video> element (works on
YouTube, Netflix, and most sites). Adjusts volume / playback speed / brightness /
captions, plays-pauses-seeks, and closes tabs.

REQUIRES a one-time Chrome setting: View → Developer → "Allow JavaScript from Apple
Events". Without it Chrome refuses `execute javascript` and these return an error.
"""
from __future__ import annotations

import subprocess

_APP = "Google Chrome"

# JS snippets — SINGLE QUOTES ONLY (they sit inside an AppleScript double-quoted
# string; a double quote here would break it). `v` = the page's main <video>.
_V = "var v=document.querySelector('video');"
_ACTIONS: dict[str, str] = {
    "volume_up":   _V + "if(v){v.muted=false;v.volume=Math.min(1,v.volume+0.15)}",
    "volume_down": _V + "if(v){v.volume=Math.max(0,v.volume-0.15)}",
    "mute":        _V + "if(v){v.muted=!v.muted}",
    "faster":      _V + "if(v){v.playbackRate=Math.min(4,v.playbackRate+0.25)}",
    "slower":      _V + "if(v){v.playbackRate=Math.max(0.25,v.playbackRate-0.25)}",
    "brighter":    _V + "if(v){var b=Math.min(2,(parseFloat(v.dataset.jb||'1'))+0.2);v.dataset.jb=b;v.style.filter='brightness('+b+')'}",
    "darker":      _V + "if(v){var b=Math.max(0.2,(parseFloat(v.dataset.jb||'1'))-0.2);v.dataset.jb=b;v.style.filter='brightness('+b+')'}",
    "captions":    _V + "var y=document.querySelector('.ytp-subtitles-button');if(y){y.click()}else if(v&&v.textTracks&&v.textTracks.length){var t=v.textTracks[0];t.mode=(t.mode==='showing'?'disabled':'showing')}",
    "pause":       _V + "if(v){v.pause()}",
    "play":        _V + "if(v){v.play()}",
    "restart":     _V + "if(v){v.currentTime=0;v.play()}",
    "forward":     _V + "if(v){v.currentTime+=10}",
    "back":        _V + "if(v){v.currentTime-=10}",
    "fullscreen":  _V + "if(v){document.fullscreenElement?document.exitFullscreen():v.requestFullscreen()}",
}
_ALIASES = {
    "louder": "volume_up", "increase": "volume_up", "up": "volume_up",
    "quieter": "volume_down", "decrease": "volume_down", "down": "volume_down",
    "speed_up": "faster", "slow_down": "slower", "subtitles": "captions", "cc": "captions",
    "skip": "forward", "rewind": "back", "resume": "play", "continue": "play",
}


class MediaError(RuntimeError):
    pass


class MediaController:
    def __init__(self, app: str = _APP):
        self._app = app

    def _osa(self, script: str) -> str:
        p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
        if p.returncode != 0:
            err = p.stderr.strip()
            if "Allow JavaScript from Apple Events" in err or "not allowed" in err.lower() or "-1743" in err:
                raise MediaError(
                    "Chrome is blocking control — enable View → Developer → "
                    "'Allow JavaScript from Apple Events' once, sir."
                )
            raise MediaError(err or "AppleScript failed")
        return p.stdout.strip()

    def _js(self, code: str) -> str:
        return self._osa(f'tell application "{self._app}" to execute active tab of front window javascript "{code}"')

    def control(self, action: str, value: str = "") -> str:
        """Run a video action on the active tab. `value` used for set_volume/set_speed."""
        a = (action or "").strip().lower().replace(" ", "_")
        a = _ALIASES.get(a, a)
        if a == "set_volume":
            lvl = max(0, min(100, int(float(value or 50))))
            self._js(_V + f"if(v){{v.muted=false;v.volume={lvl/100}}}")
            return f"video volume set to {lvl}"
        if a == "set_speed":
            rate = max(0.25, min(4.0, float(value or 1)))
            self._js(_V + f"if(v){{v.playbackRate={rate}}}")
            return f"playback speed {rate}x"
        if a not in _ACTIONS:
            raise MediaError(f"don't know how to '{action}'")
        self._js(_ACTIONS[a])
        return {"volume_up": "louder", "volume_down": "quieter", "captions": "toggled subtitles",
                "restart": "restarted from the beginning", "play": "resumed",
                }.get(a, a.replace("_", " "))

    def close_tab(self) -> str:
        self._osa(f'tell application "{self._app}" to close active tab of front window')
        return "closed the tab"
