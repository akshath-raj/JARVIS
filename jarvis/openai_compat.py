"""Helpers for talking to OpenAI-compatible chat APIs (OpenAI and Cerebras).

Backends disagree on the token-limit parameter: newer OpenAI models require
`max_completion_tokens` and reject `max_tokens`, while some providers only accept
`max_tokens`. `chat_complete` tries the modern name first and, on an "unsupported
parameter" error, swaps once and remembers the winner per endpoint — so it costs
at most one extra request the first time and none thereafter.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("jarvis.llm")

# base_url → the token param that worked ("max_completion_tokens" | "max_tokens")
_TOKEN_PARAM: dict[str, str] = {}


def chat_complete(client, *, model: str, messages: list, max_output: int | None = None,
                  temperature: float | None = None, reasoning_effort: str | None = None,
                  **kwargs):
    """Call `client.chat.completions.create`, tolerating differences between OpenAI
    and OpenAI-compatible providers: the token-param name (`max_completion_tokens`
    vs `max_tokens`), an unsupported `temperature`, and an unsupported
    `reasoning_effort` (dropped if the backend rejects it)."""
    base = str(getattr(client, "base_url", "openai"))
    token_param = _TOKEN_PARAM.get(base, "max_completion_tokens")

    def _build(tok_param: str, with_temp: bool, with_reasoning: bool) -> dict:
        args = dict(kwargs, model=model, messages=messages)
        if max_output is not None:
            args[tok_param] = max_output
        if with_temp and temperature is not None:
            args["temperature"] = temperature
        if with_reasoning and reasoning_effort is not None:
            eb = dict(args.get("extra_body") or {})
            eb["reasoning_effort"] = reasoning_effort  # gpt-oss / Cerebras: less reasoning = faster
            args["extra_body"] = eb
        return args

    with_temp = with_reasoning = True
    for _ in range(5):
        try:
            resp = client.chat.completions.create(**_build(token_param, with_temp, with_reasoning))
            _TOKEN_PARAM[base] = token_param  # cache what worked
            return resp
        except Exception as e:  # noqa: BLE001 — inspect the API error to adapt
            msg = str(e).lower()
            if "max_completion_tokens" in msg and token_param != "max_tokens":
                token_param = "max_tokens"          # older provider
            elif "max_tokens" in msg and token_param != "max_completion_tokens":
                token_param = "max_completion_tokens"  # newer OpenAI model
            elif "temperature" in msg and with_temp:
                with_temp = False                   # model only allows default temp
            elif "reasoning" in msg and with_reasoning:
                with_reasoning = False              # backend doesn't take reasoning_effort
            else:
                raise
    raise RuntimeError("chat_complete: exhausted parameter fallbacks")
