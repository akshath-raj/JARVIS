"""Client that drives the isolated browser-agent subprocess (main .venv side).

Does NOT import browser_use. Spawns `.venv-browser/bin/python -m
jarvis.browser_agent.runner`, feeds it a JSON job, and reports the result by voice.

Fire-and-report UX: `run_browser_task` returns a quick spoken ack immediately and
runs the task in the background (single-flight), then speaks the result via the
Announcer when it finishes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess

from jarvis.config import config

logger = logging.getLogger("jarvis.browser")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lock = asyncio.Lock()  # one browser task at a time (they share the Chrome profile)


def _chrome_running() -> bool:
    try:
        return subprocess.run(["pgrep", "-x", "Google Chrome"], capture_output=True).returncode == 0
    except Exception:
        return False


# ── drive the user's REAL, VISIBLE Chrome via the DevTools protocol ──────────
def _cdp_up(port: int) -> str | None:
    """Return the CDP url if a debuggable Chrome is listening on `port`, else None."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1):
            return f"http://127.0.0.1:{port}"
    except Exception:
        return None


def _launch_real_chrome_debug(port: int) -> None:
    """Launch the user's real profile, VISIBLE, with the DevTools port open so the
    agent can drive it. Restores their previous tabs."""
    subprocess.Popen(
        [config.browser_chrome_path,
         f"--remote-debugging-port={port}",
         f"--user-data-dir={config.browser_user_data_dir}",
         f"--profile-directory={config.browser_profile_dir}",
         "--restore-last-session", "--no-first-run", "--no-default-browser-check"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _ensure_real_chrome() -> tuple[str | None, str | None]:
    """Get a VISIBLE Chrome on the user's real profile we can drive; return
    (cdp_url, blocker). Blocker is a spoken message when we can't proceed."""
    import time
    port = config.browser_debug_port
    if (url := _cdp_up(port)):
        return url, None                    # already drivable → attach
    if _chrome_running():                   # open but not debuggable → must relaunch
        return None, ("Sir, please quit Google Chrome once so I can reopen it under "
                      "my control — your tabs come back, and after that I can open "
                      "and play things on your own profile.")
    _launch_real_chrome_debug(port)
    for _ in range(24):                     # wait up to ~6s for the port to come up
        if (url := _cdp_up(port)):
            return url, None
        time.sleep(0.25)
    return None, "I couldn't start Chrome with remote control, sir."


# Cache dirs (the bulk of a profile) we skip when cloning — not needed for logins.
_CLONE_EXCLUDES = [
    "*Cache*/", "*/Service Worker/CacheStorage/", "*/Service Worker/ScriptCache/",
    "*/GPUCache/", "*/DawnCache/", "*/GrShaderCache/", "ShaderCache/", "GraphiteDawnCache/",
    "Crashpad/", "Safe Browsing/", "*.log", "Singleton*", "component_crx_cache/", "extensions_crx_cache/",
]


def _clone_profile() -> str:
    """rsync the live Chrome profile (cookies/logins, minus caches) into JARVIS's own
    user-data-dir so a headless Chrome can use it WITHOUT touching the running one.
    Incremental after the first run. Returns the clone user-data-dir."""
    src = config.browser_user_data_dir.rstrip("/")
    dst = config.browser_clone_dir
    os.makedirs(dst, exist_ok=True)
    cmd = ["rsync", "-a", "--delete"]
    for e in _CLONE_EXCLUDES:
        cmd += ["--exclude", e]
    cmd += [src + "/", dst + "/"]
    subprocess.run(cmd, capture_output=True, timeout=180)
    return dst


def _effective_profile() -> tuple[str, bool, str | None]:
    """Return (user_data_dir, headless, blocker). blocker is a spoken message when we
    can't proceed (e.g. Chrome open and cloning disabled)."""
    mode = (config.browser_clone_profile or "never").lower()
    running = _chrome_running()
    if mode == "always" or (mode == "auto" and running):
        # clone the profile and drive the copy; honor the headless setting (default
        # visible) so the user can see it, instead of forcing a hidden window.
        return _clone_profile(), config.browser_headless, None
    if running:  # mode == never and Chrome is open → take over the real profile
        return config.browser_user_data_dir, config.browser_headless, \
            "Sir, please close Google Chrome first so I can open it on your own profile."
    return config.browser_user_data_dir, config.browser_headless, None


def _ollama_host() -> str:
    return config.ollama_base_url.rstrip("/").removesuffix("/v1").rstrip("/")


def _job(instruction: str, files=None, frontier_only: bool = False,
         user_data_dir: str | None = None, headless: bool | None = None,
         cdp_url: str | None = None) -> dict:
    return {
        "task": instruction,
        "cdp_url": cdp_url,  # set → attach to the user's real visible Chrome
        "chrome_path": config.browser_chrome_path,
        "user_data_dir": user_data_dir or config.browser_user_data_dir,
        "profile_directory": config.browser_profile_dir,
        "downloads_dir": config.browser_downloads,
        "headless": headless if headless is not None else config.browser_headless,
        "local_model": "" if frontier_only else (config.browser_local_model if config.browser_use_local else ""),
        "local_max_steps": config.browser_local_max_steps,
        "frontier_model": config.browser_frontier_model,
        "ollama_host": _ollama_host(),
        "openai_api_key": config.openai_api_key,
        "max_steps": config.browser_max_steps,
        "available_file_paths": files or [],
        "captcha_model": config.captcha_model if config.captcha_enabled else "",
    }


async def _run_job(instruction: str, *, files=None, frontier_only: bool, timeout: int,
                   attach: bool = False) -> dict:
    """Run the job. `attach` (interactive tasks) drives the user's real VISIBLE Chrome
    via CDP so they can see it; otherwise use the profile/clone path (background)."""
    if attach and config.browser_attach_real:
        cdp_url, blocker = await asyncio.to_thread(_ensure_real_chrome)
        if blocker:
            return {"ok": False, "result": "", "error": blocker}
        job = _job(instruction, files=files, frontier_only=frontier_only, cdp_url=cdp_url)
        return await _spawn(job, timeout)
    udd, headless, blocker = await asyncio.to_thread(_effective_profile)
    if blocker:
        return {"ok": False, "result": "", "error": blocker}
    job = _job(instruction, files=files, frontier_only=frontier_only,
               user_data_dir=udd, headless=headless)
    return await _spawn(job, timeout)


async def run_task_sync(instruction: str, *, files=None, timeout: int | None = None) -> dict:
    """Run a browser task and RETURN its result dict (no fire-and-report).

    Used by higher-level flows (download an assignment, drive claude.ai to answer)
    that need the outcome to continue. Serialised via the same single-flight lock.
    """
    async with _lock:
        return await _run_job(instruction, files=files, frontier_only=True,
                              timeout=timeout or config.browser_timeout)


async def run_browser_task(instruction: str, *, announce=None) -> str:
    """Spoken ack now; the browser task runs in the background and is announced when done."""
    if not config.browser_agent_enabled:
        return "the browser agent isn't enabled, sir"
    if _lock.locked():
        return "I'm still on the last browser task, sir — one moment"
    asyncio.create_task(_run_and_report(instruction, announce))
    return "On it, sir — I'll let you know when it's done."


async def _run_and_report(instruction: str, announce) -> None:
    async with _lock:
        result = await _execute(instruction)
    if announce is not None:
        await announce(result)
    else:
        logger.info("browser result: %s", result)


def _preflight() -> str | None:
    """Return a spoken reason we can't run, or None if good to go."""
    py = os.path.join(config.browser_venv, "bin", "python")
    if not os.path.exists(os.path.join(_REPO_ROOT, py)) and not os.path.isabs(py):
        return "the browser agent isn't set up, sir — run scripts/setup_browser_agent.sh first"
    if not config.openai_api_key:
        return "I need an OpenAI key for the browser agent, sir — set OPENAI_API_KEY"
    return None


async def _spawn(job: dict, timeout: int) -> dict:
    """Run the isolated runner subprocess; return its parsed result dict."""
    err = _preflight()
    if err:
        return {"ok": False, "result": "", "error": err}
    py = os.path.join(config.browser_venv, "bin", "python")
    try:
        proc = await asyncio.create_subprocess_exec(
            py, "-m", "jarvis.browser_agent.runner",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=_REPO_ROOT,
        )
    except OSError as e:
        return {"ok": False, "result": "", "error": f"couldn't start browser agent: {e}"}
    try:
        out, err_b = await asyncio.wait_for(proc.communicate(json.dumps(job).encode()), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"ok": False, "result": "", "error": "the browser task took too long and I stopped it"}
    lines = [ln for ln in (out or b"").decode(errors="replace").splitlines() if ln.strip()]
    if not lines:
        tail = (err_b or b"").decode(errors="replace").strip().splitlines()[-1:] or [""]
        return {"ok": False, "result": "", "error": f"no output ({tail[0][:80]})"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ok": False, "result": "", "error": "unreadable result"}


async def _execute(instruction: str) -> str:
    # interactive voice task → drive the user's real, visible Chrome
    res = await _run_job(instruction, frontier_only=False, timeout=config.browser_timeout, attach=True)
    if res.get("result"):
        return res["result"]
    if res.get("error"):
        # preflight messages are already user-facing; wrap raw errors.
        e = res["error"]
        return e if e[0:1].isupper() or e.startswith(("the ", "I ")) else f"Sorry sir, the browser task failed: {e}"
    return "The browser task finished, sir, but produced no result."
