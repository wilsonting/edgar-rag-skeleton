"""An Anthropic-shaped client backed by an OpenAI-compatible API.

The whole codebase speaks Anthropic's message shape: `tool_use` /
`tool_result` content blocks, a top-level `system`, `cache_control`
breakpoints, `stop_reason`. That shape is not incidental — the debate and
risk ports build tool-result turns by hand, the synthesis port re-sends
`response.content` verbatim on a schema retry, and the provenance guards
read block text. Rewriting all of that onto a neutral message type would
have meant touching every tested path in the pipeline.

So the internal lingua franca stays Anthropic-shaped and the translation
happens here, at the wire. This class exposes exactly the surface the
callers use — `client.messages.create(**anthropic_kwargs)` returning an
object with `.content`, `.stop_reason`, `.usage`, `.model` — and maps it
onto an OpenAI-compatible chat-completions endpoint (DeepSeek, or anything
else that speaks that dialect).

The cost of that choice, stated plainly: Anthropic-only concepts have no
target on the other side and are dropped, loudly the first time it matters
per process. See `_UNSUPPORTED` below for the full list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, BadRequestError as _OpenAIBadRequestError


class LLMBadRequestError(Exception):
    """A 400 from a non-Anthropic provider.

    Exists so `create_with_temperature_fallback` can catch a provider's
    "that parameter is not supported" the same way it catches Anthropic's,
    without importing `openai` into the debate port.
    """


# ---------------------------------------------------------------------------
# Response shim — Anthropic-shaped, so callers cannot tell the difference
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class ShimUsage:
    """Anthropic's four usage fields, filled from an OpenAI-style `usage`.

    Named with Anthropic's field names on purpose: `TokenUsage.from_response`
    and the ports' `_accumulate` both read them by those names, and a shim
    that needed special-casing at every accounting site would defeat the
    point of shimming at all.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class ShimResponse:
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: ShimUsage = field(default_factory=ShimUsage)
    model: str = ""
    id: str = ""


# OpenAI's `finish_reason` -> Anthropic's `stop_reason`. Both of the mapped
# values are load-bearing: `researcher.py` branches on "max_tokens" to
# recover a memo that was cut mid-sentence, and on "tool_use" to decide
# whether the loop continues at all.
_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def _as_dict(block: Any) -> dict[str, Any]:
    """Content blocks reach us as plain dicts (hand-built turns) or as
    objects (an earlier response replayed into the next request, which is
    what the synthesis and debate schema-retries do)."""
    if isinstance(block, dict):
        return block
    out = {"type": getattr(block, "type", None)}
    for name in ("text", "id", "name", "input", "tool_use_id", "content", "is_error"):
        if hasattr(block, name):
            out[name] = getattr(block, name)
    return out


def _flatten_text(value: Any) -> str:
    """Anthropic accepts a string OR a list of blocks almost everywhere a
    string is meaningful — `system`, and the body of a `tool_result`.
    OpenAI accepts only the string, so cache_control breakpoints and block
    boundaries collapse here."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            block = _as_dict(block)
            if block.get("type") in (None, "text"):
                parts.append(block.get("text", ""))
        return "\n\n".join(p for p in parts if p)
    return str(value)


def _translate_messages(system: Any, messages: list[dict]) -> list[dict]:
    out: list[dict] = []

    system_text = _flatten_text(system)
    if system_text:
        out.append({"role": "system", "content": system_text})

    for message in messages:
        role = message["role"]
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = [_as_dict(b) for b in (content or [])]

        if role == "assistant":
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b.get("input") or {}),
                    },
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            assistant: dict[str, Any] = {"role": "assistant"}
            joined = "\n".join(t for t in texts if t)
            # An assistant turn that is nothing but tool calls must send
            # content as null, not "" — some OpenAI-compatible servers
            # reject the empty string alongside `tool_calls`.
            assistant["content"] = joined or None
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
            continue

        # A user turn answering tools: every `tool_result` becomes its own
        # `role: "tool"` message, and they must land before any free text,
        # because the API requires each tool call to be answered
        # immediately by the message that follows its assistant turn.
        pending_text = []
        for block in blocks:
            kind = block.get("type")
            if kind == "tool_result":
                body = _flatten_text(block.get("content"))
                if block.get("is_error"):
                    body = f"ERROR: {body}"
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": body,
                    }
                )
            elif kind == "text":
                pending_text.append(block.get("text", ""))
        if pending_text:
            out.append({"role": role, "content": "\n".join(p for p in pending_text if p)})

    return out


def _translate_tools(tools: list[dict] | None) -> list[dict] | None:
    """`strict` carries over.

    It was dropped in the first version of this shim on the assumption that
    the OpenAI dialect had no target for it. DeepSeek documents the flag on
    tool definitions, and it is worth carrying: `strict` was added to the
    debate port's SUBMIT_TOOL because 3 of 3 live turns came back with a
    flattened payload without it, costing a retry every time — and a retry
    loop is the one runaway the round cap cannot see.
    """
    if not tools:
        return None
    out = []
    for tool in tools:
        function = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {},
        }
        if tool.get("strict"):
            function["strict"] = True
        out.append({"type": "function", "function": function})
    return out


def _translate_tool_choice(tool_choice: dict | str | None) -> tuple[Any, bool | None]:
    """Returns `(tool_choice, parallel_tool_calls)`.

    Anthropic folds "call this one tool, exactly once" into a single
    `tool_choice`; OpenAI splits it across two parameters.
    """
    if tool_choice is None:
        return None, None
    if isinstance(tool_choice, str):
        return tool_choice, None

    kind = tool_choice.get("type")
    parallel = False if tool_choice.get("disable_parallel_tool_use") else None
    if kind == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}, parallel
    if kind == "any":
        return "required", parallel
    if kind == "none":
        return "none", parallel
    return "auto", parallel


# DeepSeek's reasoning_effort values. Anthropic's "medium" has no
# counterpart and is dropped rather than rounded to a neighbour.
_REASONING_EFFORTS = frozenset({"low", "high", "max"})


def _translate_thinking(kwargs: dict[str, Any], forces_a_tool: bool) -> dict[str, Any] | None:
    """The provider's `thinking` object, or None to leave it at their default.

    Two things this has to get right, both learned from the live API rather
    than from the docs:

    1. Thinking is ON by default here, unlike Anthropic. A caller that sends
       no thinking parameter at all still gets it.

    2. Thinking mode REJECTS a constrained tool_choice outright — both
       `required` and a named function come back 400 "Thinking mode does not
       support this tool_choice". Only `auto` survives it.

    The ports force a named tool because their whole structured-output
    contract is "call exactly this tool, exactly once"; `_extract` raises
    when no tool block comes back, and the one retry after that raises out.
    Falling back to `auto` to keep thinking would trade a hard API guarantee
    for a behavioural hope, on the calls whose output is most load-bearing.
    So a forced tool wins and thinking is disabled for that call.

    Its reasoning tokens are billed as output either way, so nothing about
    cost accounting depends on this — only the quality of those turns.
    """
    if forces_a_tool:
        return {"type": "disabled"}

    thinking = kwargs.get("thinking")
    if not thinking:
        # Absence means OFF, and sending that explicitly is the whole point.
        #
        # The internal message shape is Anthropic's, where a call that sends
        # no `thinking` gets none. DeepSeek's default is the opposite, so
        # "leave the provider default alone" silently turned thinking ON for
        # every call that does not force a tool — and reasoning tokens are
        # billed as output and counted against `max_tokens`.
        #
        # Measured, on the news digest's own 15-article batch: 1280 output
        # tokens with thinking on against 632 with it off, into a
        # DIGEST_MAX_TOKENS of 1500. On the live ACN run all 2 digest
        # batches came back unparseable and the node refused to pass an
        # empty digest off as "no news", which ended the run; on MSFT 15 of
        # 16 batches failed and the technical node — max_tokens 512 —
        # returned an empty interpretation under its own heading.
        #
        # Faithful translation of the lingua franca fixes all of it: a
        # caller that wants reasoning here has to ask for it, exactly as it
        # would on Anthropic.
        return {"type": "disabled"}

    out: dict[str, Any] = {"type": "disabled" if thinking.get("type") == "disabled" else "enabled"}
    effort = (kwargs.get("output_config") or {}).get("effort")
    if out["type"] == "enabled" and effort in _REASONING_EFFORTS:
        out["reasoning_effort"] = effort
    return out


# Anthropic-only request parameters, and what happens to each. Warned about
# once per process rather than per call — a per-call warning on a 45-turn
# loop is noise nobody reads, and silence is how a dropped parameter turns
# into an unexplained change in behavior.
_UNSUPPORTED = {
    "betas": "beta headers are Anthropic-only; dropped",
    "top_k": "not in the OpenAI dialect; dropped",
    "metadata": "not forwarded",
}

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    print(f"[llm-shim] {message}")


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

class _Messages:
    def __init__(self, owner: "OpenAICompatClient") -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> ShimResponse:
        return await self._owner._create(**kwargs)


class OpenAICompatClient:
    """Duck-types `anthropic.AsyncAnthropic` for the surface this repo uses.

    `max_output_tokens` clamps `max_tokens` down to what the provider will
    actually accept, so a request that asks for more comes back short rather
    than 400ing at the wire. The research agent asks for 16000 because a
    12-item memo genuinely needs it, which is under every current provider's
    ceiling — but it was over the retired `deepseek-chat`'s 8192, and that
    failure would have landed on the FINAL turn of a run already paid for.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_output_tokens: int | None = None,
        provider: str = "openai-compatible",
        thinking_param: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._max_output_tokens = max_output_tokens
        self._provider = provider
        self._thinking_param = thinking_param
        self.messages = _Messages(self)

    async def _create(self, **kwargs: Any) -> ShimResponse:
        model = kwargs["model"]

        for name, note in _UNSUPPORTED.items():
            if name in kwargs:
                _warn_once(f"{self._provider}:{name}", f"{self._provider}: `{name}` — {note}")

        max_tokens = kwargs.get("max_tokens")
        if max_tokens and self._max_output_tokens and max_tokens > self._max_output_tokens:
            _warn_once(
                f"{self._provider}:max_tokens",
                f"{self._provider}: max_tokens={max_tokens} exceeds this provider's "
                f"{self._max_output_tokens} ceiling — clamping. Output long enough to "
                f"need the larger budget will come back stop_reason=max_tokens.",
            )
            max_tokens = self._max_output_tokens

        tools = _translate_tools(kwargs.get("tools"))

        tool_choice, parallel = _translate_tool_choice(kwargs.get("tool_choice"))
        # `auto` and None leave the model free to skip the tool, so they do
        # not constrain it; everything else does.
        forces_a_tool = tool_choice is not None and tool_choice != "auto"
        # Skipped entirely for a provider with no reasoning toggle — writing
        # an unknown key into extra_body is a 400 on a strict server.
        thinking = (
            _translate_thinking(kwargs, forces_a_tool) if self._thinking_param else None
        )
        if forces_a_tool and thinking == {"type": "disabled"} and kwargs.get("thinking"):
            _warn_once(
                f"{self._provider}:thinking-vs-tool_choice",
                f"{self._provider}: thinking disabled on calls that force a tool — "
                f"the API rejects the combination. The tool-call guarantee is kept; "
                f"the reasoning on those turns is not.",
            )

        request: dict[str, Any] = {
            "model": model,
            "messages": _translate_messages(kwargs.get("system"), kwargs["messages"]),
        }
        if max_tokens:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if parallel is not None and tools:
            request["parallel_tool_calls"] = parallel
        if kwargs.get("temperature") is not None:
            request["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            request["top_p"] = kwargs["top_p"]
        if kwargs.get("stop_sequences"):
            request["stop"] = kwargs["stop_sequences"]
        if thinking is not None:
            # extra_body, not a top-level kwarg: the OpenAI SDK raises
            # TypeError on parameters it does not know.
            request["extra_body"] = {self._thinking_param: thinking}

        try:
            completion = await self._client.chat.completions.create(**request)
        except _OpenAIBadRequestError as exc:
            raise LLMBadRequestError(str(exc)) from exc

        return self._to_anthropic_shape(completion, model)

    def _to_anthropic_shape(self, completion: Any, model: str) -> ShimResponse:
        choice = completion.choices[0]
        message = choice.message

        content: list[Any] = []
        if getattr(message, "content", None):
            content.append(TextBlock(text=message.content))
        for call in getattr(message, "tool_calls", None) or []:
            content.append(
                ToolUseBlock(
                    id=call.id,
                    name=call.function.name,
                    # A model can emit malformed JSON here. An empty dict
                    # loses the turn's work but reaches the callers'
                    # ValidationError retry, which is a far better outcome
                    # than a JSONDecodeError unwinding the whole run.
                    input=_safe_json(call.function.arguments),
                )
            )

        return ShimResponse(
            content=content,
            stop_reason=_STOP_REASONS.get(choice.finish_reason, "end_turn"),
            usage=_translate_usage(getattr(completion, "usage", None)),
            model=getattr(completion, "model", model) or model,
            id=getattr(completion, "id", "") or "",
        )


def _safe_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _translate_usage(usage: Any) -> ShimUsage:
    """Map an OpenAI-style `usage` onto Anthropic's four fields.

    The one thing that must not go wrong here is double-counting the cache.
    DeepSeek's `prompt_tokens` is cache hits PLUS misses, and hits are
    billed at a fraction of the miss rate. Pricing `prompt_tokens` as
    `input` and ALSO counting the hits as `cache_read` would overstate every
    cached run — so `input_tokens` is the miss count whenever the provider
    reports one, and falls back to `prompt_tokens` only when it does not
    break the number out (in which case there is no cache figure to
    double-count anyway).
    """
    if usage is None:
        return ShimUsage()

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)

    if cache_hit is None:
        # OpenAI proper reports it nested instead.
        details = getattr(usage, "prompt_tokens_details", None)
        cache_hit = getattr(details, "cached_tokens", None) if details else None

    cache_hit = cache_hit or 0
    input_tokens = cache_miss if cache_miss is not None else max(prompt - cache_hit, 0)

    return ShimUsage(
        input_tokens=input_tokens,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        # No provider in this dialect bills separately to WRITE the cache;
        # they populate it as a side effect of the miss already counted above.
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cache_hit,
    )
