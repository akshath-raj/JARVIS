"""Browser backend — drive claude.ai (the user's Pro login) to do the assignment.

Reuses the isolated browser agent: uploads the assignment file, sends the prompt,
waits for Claude to finish, and returns Claude's answer text. This is the reliable
default (browser-use reads the page + handles the upload/extraction).
"""
from __future__ import annotations

from jarvis.ai_apps.base import AnswerBackend, BackendError
from jarvis.browser_agent.client import run_task_sync
from jarvis.config import config


class BrowserBackend(AnswerBackend):
    name = "browser"

    def available(self) -> bool:
        return bool(config.browser_agent_enabled and config.openai_api_key)

    async def generate(self, assignment_path: str, prompt: str, out_format: str) -> str:
        instruction = (
            "Go to https://claude.ai and start a NEW chat. Attach/upload the file that has "
            "been made available to you. Then send Claude EXACTLY this message:\n\n"
            f"{prompt}\n\n"
            "Wait until Claude has completely finished writing its reply (it is no longer "
            "generating). Then return Claude's full and final answer VERBATIM — including all "
            "code and formatting — and nothing else."
        )
        res = await run_task_sync(
            instruction, files=[assignment_path], timeout=max(config.browser_timeout, 300)
        )
        answer = (res.get("result") or "").strip()
        if not answer:
            raise BackendError(res.get("error") or "claude.ai returned no answer")
        return answer
