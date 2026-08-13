"""FocusController — the focus-assist state machine and enforcer.

`start()` closes every open distraction (Instagram/YouTube/Netflix/… tabs, plus
any user-listed apps), begins a timed regime (Pomodoro by default), and launches
a strict watcher that keeps closing any distraction the user reopens — announcing
that focus mode is on — until `stop()` is called by voice ("end focus mode").

Everything is duck-typed and injectable so it tests offline: `closer` runs the
AppleScript that shuts distraction tabs, `announce` speaks, and the async loops
are created on the running LiveKit event loop (mirroring AlarmScheduler).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import time

logger = logging.getLogger("jarvis.focus")

# Built-in distraction domains. Substring-matched against a tab's URL, so
# "youtube.com" also catches m.youtube.com and youtube.com/shorts.
_DEFAULT_SITES: tuple[str, ...] = (
    "instagram.com", "youtube.com", "youtu.be", "netflix.com", "tiktok.com",
    "twitter.com", "x.com", "reddit.com", "facebook.com", "twitch.tv",
    "hulu.com", "disneyplus.com", "primevideo.com", "hotstar.com", "9gag.com",
    "pinterest.com", "snapchat.com", "tumblr.com", "quora.com", "discord.com",
)

# Spoken/typed names that mean a distraction, for refusing an open-by-name before
# it ever launches (the URL watcher is the backstop for anything that slips through).
_DISTRACTION_WORDS: frozenset[str] = frozenset({
    "instagram", "insta", "ig", "reels", "reel", "youtube", "yt", "shorts",
    "netflix", "tiktok", "twitter", "x", "reddit", "facebook", "fb", "twitch",
    "hulu", "disney", "disney+", "prime video", "primevideo", "hotstar", "9gag",
    "pinterest", "snapchat", "tumblr", "quora", "discord",
})

# Focus techniques: work / short-break / long-break minutes. `long_every` = take
# a long break after this many work sprints (0 = never). `flow` = open-ended, no
# forced breaks (just periodic check-ins).
TECHNIQUES: dict[str, dict] = {
    "pomodoro":  {"work": 25, "short": 5,  "long": 15, "long_every": 4, "flow": False},
    "classic":   {"work": 50, "short": 10, "long": 20, "long_every": 3, "flow": False},
    "52-17":     {"work": 52, "short": 17, "long": 17, "long_every": 0, "flow": False},
    "90-20":     {"work": 90, "short": 20, "long": 20, "long_every": 0, "flow": False},
    "flowtime":  {"work": 30, "short": 5,  "long": 5,  "long_every": 0, "flow": True},
}
_TECH_ALIASES = {
    "pomo": "pomodoro", "tomato": "pomodoro", "standard": "pomodoro",
    "fifty two seventeen": "52-17", "52/17": "52-17", "5217": "52-17",
    "ninety": "90-20", "90/20": "90-20", "ultradian": "90-20",
    "flow": "flowtime", "deep work": "classic", "deep": "classic", "desktime": "52-17",
}


def resolve_technique(name: str) -> str:
    n = (name or "").strip().lower()
    if n in TECHNIQUES:
        return n
    return _TECH_ALIASES.get(n, "pomodoro")


class FocusController:
    def __init__(self, *, announce=None, closer=None, app_closer=None, ui=None,
                 extra_sites: tuple[str, ...] = (), block_apps: tuple[str, ...] = (),
                 poll_seconds: float = 3, default_technique: str = "pomodoro",
                 minute: float = 60.0) -> None:
        self._announce = announce                         # async callable(text)
        self._ui = ui                                     # UIController (optional)
        self._closer = closer or _close_distraction_tabs  # (sites) -> list[str] hosts closed
        self._app_closer = app_closer or _quit_apps       # (apps) -> list[str] quit
        self._sites = tuple(dict.fromkeys(_DEFAULT_SITES + tuple(extra_sites)))
        self._apps = tuple(a for a in block_apps if a)
        self._poll = max(0.02, float(poll_seconds))
        self._technique = resolve_technique(default_technique)
        self._minute = minute                             # seconds per "minute" (tests shrink it)

        self.active = False
        self._cycles = 0                                  # completed work sprints
        self._phase = "idle"                              # work | break | idle
        self._phase_ends_at = 0.0
        self._started_at = 0.0
        self._last_announced_close = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []

    # ── classification (used by the browser tools to refuse a distraction) ────
    def is_distraction(self, target: str) -> bool:
        t = (target or "").strip().lower()
        if not t:
            return False
        if any(dom in t for dom in self._sites):          # a URL / domain
            return True
        return t in _DISTRACTION_WORDS or any(w == t for w in _DISTRACTION_WORDS)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    async def start(self, technique: str = "") -> str:
        """Engage focus mode: close what's open now, start the timer + watcher.

        Async so it runs ON the LiveKit event loop (the timer/watcher tasks are
        created there) while the blocking tab-sweep runs in a worker thread."""
        self._technique = resolve_technique(technique) if technique else self._technique
        closed = await asyncio.to_thread(self._sweep)     # close current distractions
        already = self.active
        self.active = True
        if not already:
            self._cycles = 0
            self._started_at = time.time()
            self._phase = "starting"
            self._phase_ends_at = 0.0
            self._start_loops()
        self._ui_push()
        tech = TECHNIQUES[self._technique]
        regime = ("open-ended flow, I'll check in periodically"
                  if tech["flow"] else f"{tech['work']}-minute sprints with {tech['short']}-minute breaks")
        shut = f" Closed {len(closed)} distraction(s): {', '.join(_pretty(closed))}." if closed else ""
        return (f"Focus mode engaged, sir — {self._technique} ({regime}).{shut} "
                "I'll shut down anything distracting until you say end focus mode.")

    def stop(self) -> str:
        if not self.active:
            return "Focus mode isn't on, sir."
        self.active = False                               # loops also poll this flag
        loop, tasks = self._loop, self._tasks
        self._tasks = []
        for t in tasks:                                   # cancel from any thread safely
            if loop is not None:
                loop.call_soon_threadsafe(t.cancel)
            else:
                t.cancel()
        self._phase = "idle"
        mins = int((time.time() - self._started_at) / self._minute) if self._started_at else 0
        self._started_at = 0.0
        self._ui_clear()
        return (f"Focus mode ended, sir. You completed {self._cycles} focus "
                f"sprint(s) over about {mins} minute(s). Well done.")

    # ── HUD push ──────────────────────────────────────────────────────────────
    def _ui_state(self) -> dict:
        return {
            "active": self.active, "technique": self._technique, "phase": self._phase,
            "ends_at": self._phase_ends_at, "cycles": self._cycles,
        }

    def _ui_push(self) -> None:
        if self._ui is not None and hasattr(self._ui, "show_focus"):
            try:
                self._ui.show_focus(self._ui_state())
            except Exception as e:
                logger.debug("focus UI push failed: %s", e)

    def _ui_clear(self) -> None:
        if self._ui is not None and hasattr(self._ui, "clear_focus"):
            try:
                self._ui.clear_focus()
            except Exception as e:
                logger.debug("focus UI clear failed: %s", e)

    def status(self) -> dict:
        remaining = max(0, int((self._phase_ends_at - time.time()) / self._minute)) if self.active else 0
        return {
            "active": self.active, "technique": self._technique, "phase": self._phase,
            "cycles": self._cycles, "minutes_left_in_phase": remaining,
            "blocked_sites": len(self._sites),
        }

    # ── enforcement ───────────────────────────────────────────────────────────
    def _sweep(self) -> list[str]:
        """Close all open distraction tabs (+ blocked apps). Returns hosts closed."""
        try:
            closed = list(self._closer(self._sites))
        except Exception as e:  # never let a shell hiccup break focus mode
            logger.warning("tab sweep failed: %s", e)
            closed = []
        if self._apps:
            try:
                closed += self._app_closer(self._apps)
            except Exception as e:
                logger.warning("app sweep failed: %s", e)
        return closed

    def _start_loops(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("no running loop; focus loops not started")
            return
        self._tasks = [
            self._loop.create_task(self._watch_loop()),
            self._loop.create_task(self._timer_loop()),
        ]

    async def _watch_loop(self) -> None:
        """Keep closing distractions the user reopens, announcing focus is on."""
        while self.active:
            closed = await asyncio.to_thread(self._sweep_quiet)
            if closed:
                now = time.time()
                # rate-limit so repeated reopen attempts don't spam the same line
                if now - self._last_announced_close > 12:
                    self._last_announced_close = now
                    await self._say(
                        f"{', '.join(_pretty(closed))} is off-limits — closing it. "
                        "We're in focus mode, sir."
                    )
            await asyncio.sleep(self._poll)

    def _sweep_quiet(self) -> list[str]:
        try:
            return list(self._closer(self._sites))
        except Exception as e:
            logger.debug("watch sweep failed: %s", e)
            return []

    async def _timer_loop(self) -> None:
        """Run the work/break regime for the chosen technique until stopped."""
        tech = TECHNIQUES[self._technique]
        try:
            while self.active:
                # ── work sprint ──
                work = tech["work"]
                await self._say(
                    f"Sprint {self._cycles + 1}: {work} minutes of focus. Begin, sir."
                    if not tech["flow"] else
                    f"Deep focus block started — I'll check in every {work} minutes, sir."
                )
                if not await self._phase_sleep("work", work):
                    return
                self._cycles += 1

                if tech["flow"]:
                    await self._say(
                        f"{work * self._cycles} minutes of focus done, sir — "
                        "keep going, or say take a break."
                    )
                    continue

                # ── break ──
                long = tech["long_every"] and self._cycles % tech["long_every"] == 0
                brk = tech["long"] if long else tech["short"]
                await self._say(
                    f"{'Long break' if long else 'Break'}: {brk} minutes. Step away, sir."
                )
                if not await self._phase_sleep("break", brk):
                    return
                await self._say("Break over — back to focus, sir.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("focus timer error: %s", e)

    async def _phase_sleep(self, phase: str, minutes: float) -> bool:
        """Sleep out a phase in small steps; return False if focus mode stopped."""
        self._phase = phase
        self._phase_ends_at = time.time() + minutes * self._minute
        self._ui_push()   # refresh the HUD countdown target on each phase change
        while self.active and time.time() < self._phase_ends_at:
            await asyncio.sleep(min(self._minute, max(0.05, self._phase_ends_at - time.time())))
        return self.active

    async def _say(self, text: str) -> None:
        if self._announce is None:
            logger.info("(focus) %s", text)
            return
        try:
            await self._announce(text)
        except Exception as e:
            logger.debug("focus announce failed: %s", e)


# ── macOS side-effects (AppleScript) ────────────────────────────────────────
def _pretty(hosts: list[str]) -> list[str]:
    """'youtube.com' → 'YouTube'; de-duped, order-preserved, capped for speech."""
    seen, out = set(), []
    for h in hosts:
        name = h.split(".")[0] if "." in h else h
        label = {"youtube": "YouTube", "youtu": "YouTube", "x": "X", "9gag": "9GAG",
                 "tiktok": "TikTok", "primevideo": "Prime Video",
                 "disneyplus": "Disney+"}.get(name, name.capitalize())
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out[:5]


def _run_osa(script: str, timeout: int = 15) -> str:
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "osascript failed")
    return p.stdout.strip()


def _close_distraction_tabs(sites: tuple[str, ...], app: str = "Google Chrome") -> list[str]:
    """Close every Chrome tab whose URL matches a blocked domain. Returns the list
    of matched hosts closed. Does nothing (and never launches Chrome) if Chrome
    isn't already running."""
    blocked = "{" + ", ".join(f'"{s}"' for s in sites) + "}"
    script = (
        'tell application "System Events" to set isRunning to (exists process "' + app + '")\n'
        'if not isRunning then return ""\n'
        f'tell application "{app}"\n'
        f'  set blocked to {blocked}\n'
        '  set closedHosts to ""\n'
        '  repeat with w in windows\n'
        '    set idxs to {}\n    set ti to 0\n'
        '    repeat with t in tabs of w\n'
        '      set ti to ti + 1\n'
        '      try\n        set u to URL of t\n      on error\n        set u to ""\n      end try\n'
        '      repeat with b in blocked\n'
        '        if u contains b then\n'
        '          set idxs to idxs & ti\n'
        '          set closedHosts to closedHosts & b & ","\n'
        '          exit repeat\n'
        '        end if\n'
        '      end repeat\n'
        '    end repeat\n'
        '    set n to (count of idxs)\n'
        '    repeat with i from n to 1 by -1\n'
        '      try\n        close tab (item i of idxs) of w\n      end try\n'
        '    end repeat\n'
        '  end repeat\n'
        '  return closedHosts\n'
        'end tell'
    )
    out = _run_osa(script)
    return [h for h in out.split(",") if h]


def _quit_apps(apps: tuple[str, ...]) -> list[str]:
    """Quit the named macOS apps if running. Returns those actually quit."""
    quit_now: list[str] = []
    for a in apps:
        try:
            running = _run_osa(
                f'tell application "System Events" to (exists process "{a}")'
            )
            if running == "true":
                _run_osa(f'tell application "{a}" to quit', timeout=8)
                quit_now.append(a)
        except Exception as e:
            logger.debug("couldn't quit %s: %s", a, e)
    return quit_now
