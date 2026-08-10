"""VoiceLLMAdapter — a LiveKit LLM that runs a LangGraph graph, spoken-safe.

LiveKit's stock `langchain.LLMAdapter` streams *every* message token from the
graph to TTS — including `ToolMessage` outputs and (with a multi-agent supervisor)
handoff plumbing like "Transferring back to supervisor". For a voice assistant that
means JARVIS speaks the raw tool return AND its own summary, doubled up.

We only ever want the model's final natural-language reply spoken. In LangGraph's
`stream_mode="messages"` output, the model's tokens arrive as `AIMessageChunk`
(a `BaseMessageChunk`); tool results arrive as full `ToolMessage`s (not chunks).
So filtering to `BaseMessageChunk` tokens keeps the spoken reply and drops the
tool/plumbing noise. Verified end-to-end during the feasibility spike.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessageChunk
from livekit.agents import llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.plugins.langchain.langgraph import (
    LangGraphStream,
    LLMAdapter,
    _extract_message_chunk,
    _to_chat_chunk,
)


_MODES = ["messages", "custom"]


class _VoiceGraphStream(LangGraphStream):
    async def _run(self) -> None:
        state = self._chat_ctx_to_state()
        try:
            aiter = self._graph.astream(
                state,
                self._config,
                context=self._context,
                stream_mode=_MODES,
                subgraphs=self._subgraphs,
            )
        except TypeError:  # older LangGraph without context/subgraphs kwargs
            aiter = self._graph.astream(state, self._config, stream_mode=_MODES)

        async for item in aiter:
            # With multiple stream modes LangGraph yields (mode, data).
            if isinstance(item, tuple) and len(item) == 2 and item[0] in _MODES:
                mode, data = item
                if mode == "custom":
                    # The "speak" node's tool-result text — speak it verbatim.
                    chunk = _to_chat_chunk(data)
                    if chunk:
                        self._event_ch.send_nowait(chunk)
                    continue
                token = _extract_message_chunk(data)
            else:
                token = _extract_message_chunk(item)
            # In "messages" mode, only the model's streamed tokens are
            # BaseMessageChunk. Drop ToolMessages / full messages so TTS never
            # speaks raw tool output.
            if token is None or not isinstance(token, BaseMessageChunk):
                continue
            chunk = _to_chat_chunk(token)
            if chunk:
                self._event_ch.send_nowait(chunk)


class VoiceLLMAdapter(LLMAdapter):
    """Drop-in replacement for `langchain.LLMAdapter` that only speaks the model's
    final reply (filters tool/handoff messages out of the TTS stream)."""

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> _VoiceGraphStream:
        return _VoiceGraphStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            graph=self._graph,
            conn_options=conn_options,
            config=self._config,
            context=self._context,
            subgraphs=self._subgraphs,
            stream_mode="messages",
        )
