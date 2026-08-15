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
from jarvis.config import config
from jarvis.openai_agent.brain import BrainContext

logger = logging.getLogger("jarvis.openai_agent.adapter")


def _chat_ctx_to_input(chat_ctx: ChatContext) -> list[dict]:
    """Convert the LiveKit chat context to Agents SDK input items.

    We forward the user/assistant turns (the agents carry their own system prompt
    via `instructions`, so LiveKit-layer system messages are dropped). Turns before
    the most recent long silence are dropped — a guaranteed backstop so a command
    can't be answered in the context of a minutes-old, unrelated conversation even
    if the LiveKit context wasn't pruned upstream."""
    from jarvis.activation import current_conversation

    src = chat_ctx.items
    gap = config.conversation_gap_seconds
    if gap > 0:
        src = current_conversation(src, max_gap=gap)
    items: list[dict] = []
    for it in src:
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
# gpt-oss "harmony" control tokens + stray tool-call tags that sometimes leak into
# the assistant's spoken text instead of being parsed as a real tool call. A control
# token is often followed by a bare channel/role word (analysis/commentary/final/…);
# strip that too so it isn't spoken.
_CTRL = re.compile(
    r"<\|[^|>]*\|>\s*(?:analysis|commentary|final|assistant|developer|system|user)?",
    re.I,
)
_TOOLTAG = re.compile(r"</?(tool_call|function_call|function|tool)\b[^>]*>", re.I)
# gpt-oss/harmony tool ROUTING syntax that sometimes leaks as spoken text:
#   "commentary to=functions.play_song", "functions.change_volume", "to=functions.x".
# This is tool-call plumbing, never something to speak — strip the whole token.
_FUNCROUTE = re.compile(r"\b(?:to\s*=\s*)?functions?\.[A-Za-z_]\w*", re.I)
# Keys that mark a leaked tool-call JSON object (e.g. {"tool":"take_screenshot", …}).
_TOOL_KEYS = ('"tool"', '"name"', '"function"', '"arguments"', '"parameters"', '"recipient"')


def _strip_tool_json(text: str) -> str:
    """Remove any balanced {...} object that looks like a leaked tool call, leaving
    the surrounding prose intact (the model sometimes prints the call as text)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] == "{":
            depth, j = 0, i
            while j < n:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n and any(k in text[i:i + 48] for k in _TOOL_KEYS):
                i = j + 1                       # drop the whole tool-call object
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _clean(text: str) -> str:
    t = _strip_tool_json(text or "")
    t = _CTRL.sub(" ", t)
    t = _TOOLTAG.sub(" ", t)
    t = _FUNCROUTE.sub(" ", t)
    t = _MD.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _output_text(value: object) -> str:
    """Normalise an SDK/tool result into text safe for the voice channel.

    Function tools normally return strings, but the Agents SDK also permits a
    ``ToolOutputText`` object.  Treating both forms explicitly avoids a blank
    (or a Pydantic repr) reaching the speech layer.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else getattr(value, "text", value)
    return _clean(str(text).strip())


# A final reply that is pure acknowledgement with no content of its own. When the
# model emits one of these AFTER an explain/summarise/review tool produced a real
# answer, it has lazily thrown the answer away — we speak the tool's answer instead.
_FILLER_REPLY = re.compile(
    r"^(there you (go|have it)|here (you go|it is)|as requested|all done|done|"
    r"that'?s (the|your) (summary|explanation|answer|review)|"
    r"i'?ve (explained|summari[sz]ed|reviewed)( it| that)?|explained|reviewed|summari[sz]ed)"
    r"[\s,.!]*(sir[\s,.!]*)?$",
    re.I,
)


def _longest_tool_output(result: object) -> str:
    """The longest substantial tool result in the run (the real answer, usually)."""
    best = ""
    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", None) != "tool_call_output_item":
            continue
        out = _output_text(getattr(item, "output", None))
        if len(out) > len(best):
            best = out
    return best


def _spoken_result(result: object) -> str:
    """Return a voice response even if the SDK omits ``final_output``.

    The normal fast-path ends a device-action run at the tool result. Some
    provider/SDK combinations record that result in ``new_items`` without also
    setting ``final_output``. The action has happened, but the old adapter then
    sent no chunks at all. Recover the latest actual tool output deterministically.
    Also guard the opposite failure: the model replacing a real explanation with a
    hollow "there you have it, sir." — speak the substantial tool answer instead.
    """
    text = _output_text(getattr(result, "final_output", None))
    if text:
        if _FILLER_REPLY.match(text.strip()):
            best = _longest_tool_output(result)
            if len(best) > 120:  # a real answer the model tried to shrug off
                logger.info("Model returned filler over a substantial tool answer; speaking the answer")
                return best
        return text
    best = _longest_tool_output(result)
    if best:
        logger.warning("Runner returned no final_output; speaking the completed tool result")
        return best
    logger.warning("Runner completed without a final response or tool output")
    return "Done, sir."


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
        ctx = BrainContext(memory=self._memory, memory_block=block, user_text=last_user)

        request_id = str(uuid4())
        # Run to completion and speak ONLY the final answer — we deliberately do NOT
        # stream token deltas. The brain is a reasoning model (gpt-oss) in a
        # multi-tool loop: between/around tool calls it can emit reasoning and tool
        # PREAMBLES as assistant text ("now the user will provide us input"), and
        # streaming every delta pushed that plumbing straight to TTS. `final_output`
        # is the clean final message (or a stop-tool's return string), so voicing
        # just that keeps intermediate thinking out of the speaker.
        try:
            result = await Runner.run(
                self._brain, input=items or last_user or "", context=ctx, max_turns=8
            )
            text = _spoken_result(result)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Tool execution can finish before an upstream provider fails while
            # resolving the final response. Never leave the user with a completed
            # action and no audible acknowledgement.
            logger.exception("OpenAI agent run failed before a voice response")
            text = "I completed the action, but had trouble preparing my reply, sir."
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
