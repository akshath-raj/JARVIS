"""Isolated browser-use runner — runs UNDER .venv-browser, out-of-process.

Reads one JSON job from stdin, drives the user's real Chrome profile with
browser-use, and prints exactly one JSON result line to stdout. Hybrid brain:
try the LOCAL model first (DOM-only, cheap/private); if it doesn't clearly
succeed, escalate to a FRONTIER model (with vision). This is the ONLY module that
imports `browser_use`, so its heavy/conflicting deps never touch the main app.

Job (stdin JSON):
  task, chrome_path, user_data_dir, profile_directory, downloads_dir,
  local_model, frontier_model, ollama_host, openai_api_key,
  max_steps, local_max_steps
Result (stdout JSON):
  {ok, result, steps, model_used, error}
"""
from __future__ import annotations

import asyncio
import json
import os
import sys


def _log(msg: str) -> None:
    # Diagnostics go to stderr so stdout stays a single clean JSON line.
    print(f"[browser-runner] {msg}", file=sys.stderr, flush=True)


async def _attempt(task: str, llm, browser, *, use_vision: bool, max_steps: int, files=None, tools=None) -> dict:
    from browser_use import Agent

    kwargs = {"tools": tools} if tools is not None else {}
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=use_vision,
        use_thinking=False,   # snappier; we don't need CoT narration
        available_file_paths=files or [],  # files the agent is allowed to upload
        **kwargs,
    )
    history = await agent.run(max_steps=max_steps)
    ok = history.is_successful()  # True / False / None(not done)
    return {
        "ok": bool(ok),
        "result": (history.final_result() or "").strip(),
        "steps": history.number_of_steps(),
    }


async def _run(job: dict) -> dict:
    from browser_use import Browser, ChatOllama, ChatOpenAI

    task = job["task"]
    max_steps = int(job.get("max_steps", 40))
    local_max = int(job.get("local_max_steps", 15))
    files = [os.path.expanduser(f) for f in (job.get("available_file_paths") or [])]
    result = {"ok": False, "result": "", "steps": 0, "model_used": None, "error": None}

    # Optional CAPTCHA solving via a LOCAL vision model (adds a solve_captcha action).
    tools = None
    if job.get("captcha_model"):
        try:
            from jarvis.browser_agent.captcha import build_captcha_tools

            tools = build_captcha_tools(job["captcha_model"], job.get("ollama_host", "http://localhost:11434"))
            _log(f"captcha solving enabled (local {job['captcha_model']})")
        except Exception as e:  # noqa: BLE001
            _log(f"captcha tools unavailable: {e}")

    headless = bool(job.get("headless", False))
    cdp_url = job.get("cdp_url")
    if cdp_url:
        # Attach to the user's already-running, VISIBLE Chrome (real profile). We
        # never launched it, so we never close it — the user keeps their browser.
        _log(f"attaching to real Chrome via CDP {cdp_url}")
        browser = Browser(cdp_url=cdp_url)
    else:
        browser = Browser(
            executable_path=job["chrome_path"],
            user_data_dir=os.path.expanduser(job["user_data_dir"]),
            profile_directory=job.get("profile_directory", "Default"),
            headless=headless,
            # a VISIBLE launched run stays open so the user can keep using it;
            # only a headless/background run is torn down below.
            keep_alive=not headless,
            accept_downloads=True,
            downloads_path=os.path.expanduser(job.get("downloads_dir", "~/Downloads")),
        )
    try:
        # 1) Local attempt (private, no vision).
        if job.get("local_model"):
            try:
                local = ChatOllama(
                    model=job["local_model"],
                    host=job.get("ollama_host", "http://localhost:11434"),
                )
                _log(f"local attempt with {job['local_model']}")
                r = await _attempt(task, local, browser, use_vision=False, max_steps=local_max, files=files, tools=tools)
                result.update(r, model_used="local")
            except Exception as e:  # noqa: BLE001
                _log(f"local attempt errored: {e}")
                result["error"] = f"local: {type(e).__name__}: {e}"

        # 2) Escalate to the frontier model if local didn't clearly succeed.
        if not result["ok"] and job.get("openai_api_key"):
            os.environ["OPENAI_API_KEY"] = job["openai_api_key"]
            frontier = ChatOpenAI(model=job.get("frontier_model", "gpt-4.1-mini"))
            _log(f"escalating to frontier {job.get('frontier_model', 'gpt-4.1-mini')}")
            r = await _attempt(task, frontier, browser, use_vision=True, max_steps=max_steps, files=files, tools=tools)
            result.update(r, model_used="frontier", error=None if r["ok"] else result["error"])
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        # Never close the user's attached Chrome or a visible launched window; only
        # tear down a headless/background browser we launched ourselves.
        if not cdp_url and headless:
            for closer in ("stop", "close", "kill"):
                fn = getattr(browser, closer, None)
                if callable(fn):
                    try:
                        res = fn()
                        if asyncio.iscoroutine(res):
                            await res
                        break
                    except Exception:  # noqa: BLE001
                        continue
    return result


def main() -> None:
    try:
        job = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "result": "", "error": f"bad job json: {e}"}))
        return
    if not job.get("task"):
        print(json.dumps({"ok": False, "result": "", "error": "no task"}))
        return
    try:
        result = asyncio.run(_run(job))
    except Exception as e:  # noqa: BLE001
        result = {"ok": False, "result": "", "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
