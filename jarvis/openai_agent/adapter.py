"""OpenAIAgentsLLM — a LiveKit `llm.LLM` that runs the OpenAI Agents SDK brain.

This is the seam the user asked for: the whole agentic framework (triage + handoffs
+ specialist tools) lives HERE, plugged into the LiveKit voice pipeline as the LLM
node. STT and TTS stay exactly as configured by `pipeline.py`; only the "brain" in
the middle is our custom Agents SDK runtime.

Each turn:
  1. LiveKit hands us the running `ChatContext`.
  2. We convert it to Agents-SDK input items and inject the most relevant long-term
     memories (personalisation, matching the LangGraph brain).
  3. `Runner.run` executes the agent (which may call several tools in sequence) and
     we speak ONLY its final answer — never intermediate reasoning or tool preambles.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from uuid import uuid4

from agents import Runner
from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from jarvis.openai_agent.brain import BrainContext

logger = logging.getLogger("jarvis.openai_agent.adapter")


def _chat_ctx_to_input(chat_ctx: ChatContext) -> list[dict]:
    """Convert the LiveKit chat context to Agents SDK input items.

    We forward the user/assistant turns (the agents carry their own system prompt
    via `instructions`, so LiveKit-layer system messages are dropped)."""
    items: list[dict] = []
    for it in chat_ctx.items:
        role = getattr(it, "role", None)
        if role not in ("user", "assistant"):
            continue
        text = (getattr(it, "text_content", None) or "").strip()
        if text:
            items.append({"role": role, "content": text})
    return items


def _last_user_text(items: list[dict]) -> str:
    for it in reversed(items):
        if it["role"] == "user":
            return it["content"]
    return ""


# strip markdown emphasis/heading/code characters so TTS never speaks "asterisk"
_MD = re.compile(r"[*_`#]+")


def _clean(text: str) -> str:
    return _MD.sub("", text or "")


class _AgentsStream(llm.LLMStream):
    def __init__(self, adapter: "OpenAIAgentsLLM", *, chat_ctx, tools, conn_options) -> None:
        super().__init__(adapter, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._brain = adapter._brain
        self._memory = adapter._memory

    async def _run(self) -> None:
        items = _chat_ctx_to_input(self._chat_ctx)
        last_user = _last_user_text(items)

        # Personalise: pull the memories most relevant to this turn into the triage
        # agent's dynamic instructions (via the run context).
        try:
            block = await asyncio.to_thread(self._memory.prompt_block, last_user, 5)
        except Exception:  # memory is best-effort; never block a reply
            block = ""
        ctx = BrainContext(memory=self._memory, memory_block=block)

        request_id = str(uuid4())
        # Run to completion and speak ONLY the final answer — we deliberately do NOT
        # stream token deltas. The brain is a reasoning model (gpt-oss) in a
        # multi-tool loop: between/around tool calls it can emit reasoning and tool
        # PREAMBLES as assistant text ("now the user will provide us input"), and
        # streaming every delta pushed that plumbing straight to TTS. `final_output`
        # is the clean final message (or a stop-tool's return string), so voicing
        # just that keeps intermediate thinking out of the speaker.
        result = await Runner.run(
            self._brain, input=items or last_user or "", context=ctx, max_turns=8
        )
        final = getattr(result, "final_output", None)
        text = _clean(str(final).strip()) if final is not None else ""
        if text:
            self._event_ch.send_nowait(
                ChatChunk(id=request_id, delta=ChoiceDelta(role="assistant", content=text))
            )


class OpenAIAgentsLLM(llm.LLM):
    """LiveKit LLM node backed by the OpenAI Agents SDK triage-and-handoff brain."""

    def __init__(self, *, brain, memory) -> None:
        super().__init__()
        self._brain = brain
        self._memory = memory

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> _AgentsStream:
        # The agents own their tools; LiveKit-registered tools are ignored here.
        return _AgentsStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)
