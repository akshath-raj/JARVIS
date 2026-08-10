"""Assignment workflows: download+explain, do-assignment, open-answer.

These are the long-running background jobs behind the graph tools. Each speaks its
result via `announce` (fire-and-report), so the voice loop never blocks.
"""
from __future__ import annotations

import logging
import os
import time

from jarvis.ai_apps import BackendError, available_backends
from jarvis.ai_apps.prompts import build_prompt
from jarvis.browser_agent.client import run_task_sync
from jarvis.config import config
from jarvis.documents.assemble import assemble, detect_format
from jarvis.documents.explain import explain
from jarvis.documents.openwith import open_with_app
from jarvis.documents.parse import DocumentError
from jarvis.documents.workspace import Workspace

logger = logging.getLogger("jarvis.documents")


async def _say(announce, text: str) -> None:
    if announce is not None:
        await announce(text)
    else:
        logger.info("(no announce) %s", text)


async def download_and_explain(what: str, workspace: Workspace, announce=None) -> None:
    start = time.time() - 5  # small skew so a just-finished download counts
    res = await run_task_sync(
        f"Download {what}. Log in if needed and save the file to the Downloads folder.",
        timeout=max(config.browser_timeout, 240),
    )
    if not res.get("ok") and res.get("error"):
        await _say(announce, res["error"] if res["error"][0:1].isupper() else
                   f"Sorry sir, I couldn't download it: {res['error']}")
        return
    path = workspace.newest_document(since=start)
    if not path:
        await _say(announce, "I ran the download, sir, but couldn't find the file in Downloads.")
        return
    name = os.path.basename(path)
    try:
        explanation = await _to_thread(explain, path)
    except DocumentError as e:
        workspace.set_assignment(path)
        await _say(announce, f"I downloaded {name}, sir, but couldn't read it: {e}")
        return
    workspace.set_assignment(path, explanation)
    await _say(announce, f"I've downloaded {name}, sir. {explanation}")


async def do_assignment(instructions: str, workspace: Workspace, announce=None) -> None:
    if not workspace.current_assignment:
        await _say(announce, "There's no assignment loaded, sir — ask me to download one first.")
        return
    path = workspace.current_assignment
    out_format = detect_format(instructions, workspace.explanation or "")
    prompt = build_prompt(instructions, out_format, os.path.basename(path))

    backends = available_backends()
    if not backends:
        await _say(announce, "I've no AI available to do it, sir — set an OpenAI key or enable Claude.")
        return
    answer, used, last_err = "", None, None
    for backend in backends:
        try:
            answer = await backend.generate(path, prompt, out_format)
            used = backend.name
            break
        except BackendError as e:
            last_err = str(e)
            logger.warning("backend %s failed: %s", backend.name, e)
    if not answer:
        await _say(announce, f"Sorry sir, I couldn't get the assignment done: {last_err or 'unknown error'}")
        return
    try:
        answer_path = assemble(answer, out_format, config.answers_dir)
    except Exception as e:  # noqa: BLE001
        await _say(announce, f"I got the answer, sir, but couldn't save the file: {e}")
        return
    workspace.set_answer(answer_path)
    await _say(
        announce,
        f"Done, sir — I've prepared {os.path.basename(answer_path)} using {used}. "
        "Shall I open it so you can review?",
    )


def open_answer(workspace: Workspace) -> str:
    if not workspace.answer_file or not os.path.exists(workspace.answer_file):
        return "There's no finished answer to open yet, sir."
    try:
        return open_with_app(workspace.answer_file).replace("opened", "Opening") + ", sir."
    except Exception as e:  # noqa: BLE001
        return f"Sorry sir, I couldn't open it: {e}"


async def _to_thread(fn, *a):
    import asyncio

    return await asyncio.to_thread(fn, *a)
