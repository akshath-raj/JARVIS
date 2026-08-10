"""Specialist engine — a reusable, self-contained "mini-agent".

A `Specialist` is a lean domain prompt + a small set of SCOPED tools + its own
local tool-calling loop against Ollama (OpenAI-compatible `/v1`). The supervisor
(the LiveKit voice agent) delegates a request to a specialist by calling
`specialist.run(request)` inside a `@function_tool`.

Why our own loop instead of a LiveKit agent handoff? On local models, swapping
the session's Agent makes the model narrate the tool call as plain text instead
of executing it. Here there is NO agent swap: each specialist runs a fresh,
focused inference over only its own tools — the exact setup this codebase already
proved reliable — so tool calls actually execute.

Each specialist sees only its own tools, so tool selection stays reliable even as
the overall system grows to many domains. Add a new domain = add a new Specialist
subclass + register one delegation tool on the supervisor.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

logger = logging.getLogger("jarvis.specialist")


@dataclass
class ToolSpec:
    """One scoped tool: a JSON-schema contract + the callable that runs it.

    `func` is a plain Python callable returning a short result string (what the
    supervisor will ultimately speak). It may raise — the loop turns exceptions
    into an ``error: ...`` result so one bad call never crashes the voice loop.
    """

    name: str
    description: str
    parameters: dict  # JSON schema for the function arguments
    func: Callable[..., str]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def no_params() -> dict:
    """JSON schema for a tool that takes no arguments."""
    return {"type": "object", "properties": {}, "required": []}


class Specialist:
    """Base focused sub-agent. Subclasses set `name`/`instructions` and pass their
    `ToolSpec`s to `__init__`.

    Set `chain=True` for domains that need multi-step tool sequences (later tool
    calls depend on earlier results). The default (`chain=False`) executes the
    model's tool call(s) in a single round and returns immediately — one LLM hop,
    which covers every current music/browser command (compound requests arrive as
    multiple tool_calls in that one round).
    """

    name: str = "specialist"
    instructions: str = ""
    chain: bool = False

    def __init__(self, client: OpenAI, model: str, tools: list[ToolSpec]):
        self._client = client
        self._model = model
        self._tools = {t.name: t for t in tools}

    def _schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def run(self, request: str, max_rounds: int = 4) -> str:
        """Route `request` to the right scoped tool(s) and return a short result."""
        messages: list[dict] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": request},
        ]
        last_result = ""
        for _ in range(max_rounds):
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._schemas(),
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                # No tool wanted — the model answered directly.
                return (msg.content or last_result or "").strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )
            results = []
            for c in calls:
                result = self._dispatch(c.function.name, c.function.arguments)
                results.append(result)
                messages.append(
                    {"role": "tool", "tool_call_id": c.id, "content": result}
                )
            last_result = "; ".join(results)
            if not self.chain:
                return last_result
        return last_result or "Done."

    def _dispatch(self, name: str, arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name}"
        try:
            kwargs = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            kwargs = {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        try:
            return str(tool.func(**kwargs))
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"
        except Exception as e:  # domain error (Spotify/browser/network) — stay alive
            logger.warning("tool %s failed: %s", name, e)
            return f"error: {e}"
