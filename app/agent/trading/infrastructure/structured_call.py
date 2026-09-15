"""One forced tool call, one schema retry, cost accounted — for every port.

The debate, risk and synthesis ports each carried their own copy of the
same six helpers. `_accumulate` was byte-identical in all three; `_extract`,
`_tool_block`, `_retry_messages`, `_CORRECTION` and the schema-retry loop
differed only in which payload class and tool name they named, and the
synthesis versions had already been generalised to take both as arguments.
`_assert_within_budget` existed four times, counting the news digest.

That is not a style problem. All three ports implement the same contract —
"call exactly this tool, exactly once; on a schema violation feed the error
back and try once more; then raise" — and four separate implementations of
one contract means four places to keep a fix, and four chances to keep
three of them. The per-port pieces that genuinely differ (which model,
which tool, what reasoning config, what ceiling) stay in the ports, passed
in as arguments.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Awaitable, Callable, NoReturn

from pydantic import ValidationError

from app.agent.researcher import UsageSummary
from app.agent.trading.domain.budget import NodeBudgetExceeded

CORRECTION = (
    "That submission did not validate:\n{error}\n\n"
    "Call {tool} once more, correcting exactly those fields. "
    "Change nothing else."
)


def accumulate(usage: UsageSummary, raw) -> None:
    """Add one response's tokens to a running total."""
    usage.input_tokens += raw.input_tokens
    usage.cache_write_tokens += raw.cache_creation_input_tokens or 0
    usage.cache_read_tokens += raw.cache_read_input_tokens or 0
    usage.output_tokens += raw.output_tokens


def tool_block(response):
    return next((b for b in response.content if b.type == "tool_use"), None)


def extract(response, payload_cls, tool_name: str):
    """The validated payload.

    Raises ValidationError on anything the retry can correct — including a
    response with no tool call at all, which is the same class of failure as
    a malformed one and gets the same single retry.
    """
    block = tool_block(response)
    if block is None:
        raise ValidationError.from_exception_data(
            payload_cls.__name__,
            [{"type": "missing", "loc": (tool_name,), "input": None}],
        )
    return payload_cls.model_validate(block.input)


def retry_messages(
    messages: list[dict], response, error: Exception, tool_name: str
) -> list[dict]:
    """Feed the validation error back in a shape the API accepts.

    The correction rides on a `tool_result` for the offending call when
    there was one; a response that made no tool call at all has no id to
    attach it to, so it goes as plain user text instead.
    """
    turns = list(messages)
    if response.content:
        turns.append({"role": "assistant", "content": response.content})
    correction = CORRECTION.format(error=error, tool=tool_name)
    results = [
        {
            "type": "tool_result",
            "tool_use_id": block.id,
            "is_error": True,
            "content": correction,
        }
        for block in response.content
        if block.type == "tool_use"
    ]
    turns.append({"role": "user", "content": results or correction})
    return turns


async def call_with_schema_retry(
    submit: Callable[[list[dict]], Awaitable[Any]],
    *,
    payload_cls,
    tool_name: str,
    messages: list[dict],
    usage: UsageSummary,
    label: str,
):
    """One forced tool call, and exactly one retry on a schema violation.

    `submit(messages)` is the port's own call — its model, its tool, its
    reasoning config. Everything around it is the same everywhere.

    Exactly one retry, never a loop: retries inside a node are invisible to
    the checkpointer, so an unbounded retry is a runaway no round cap can
    see — it lives entirely inside one super-step. One retry, then raise,
    then resume from the checkpoint.

    Both attempts are accounted whether or not the second succeeds. The
    first was billed regardless.
    """
    response = await submit(messages)
    accumulate(usage, response.usage)
    try:
        return extract(response, payload_cls, tool_name)
    except ValidationError as first:
        block = tool_block(response)
        print(
            f"{label}: schema violation, one retry — "
            f"stop_reason={response.stop_reason} "
            f"keys={sorted(block.input) if block else None} "
            f"— {'; '.join(str(first).splitlines()[1:5])}"
        )
        retried = retry_messages(messages, response, first, tool_name)
        retry = await submit(retried)
        accumulate(usage, retry.usage)
        return extract(retry, payload_cls, tool_name)   # a second failure raises out


def force_crash(label: str, detail: str) -> NoReturn:
    """Kill the process the way a real crash does.

    `os._exit`, not `sys.exit` or `raise`. Both of those unwind cleanly and
    let the framework write a shutdown checkpoint, which is a strictly
    easier scenario than the process kill the crash-resume exit criterion
    describes.

    The flushes are not optional: `os._exit` skips every atexit hook and
    every buffered stream, so without them the crash message and the whole
    run's node progress are lost — a faithful simulation of `kill -9` and a
    useless test log.
    """
    print(f"[{label}] FORCED CRASH {detail}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def crash_marker(at_env: str, when_env: str | None = None) -> tuple[str | None, str]:
    """The configured crash point, read once at import as the ports do."""
    return os.getenv(at_env), os.getenv(when_env or "", "before") or "before"


def assert_within_budget(
    total_usd: float,
    ceiling_usd: float,
    *,
    what: str,
    budget: str,
    check: str,
    context: str = "",
) -> None:
    """Raise NodeBudgetExceeded when a node's own spend crosses its ceiling.

    Fires as soon as the running total crosses, not at the end: an assertion
    cannot refund a call already paid for, so the earliest possible point is
    the only useful one. What it actually catches is prompt bloat or a model
    routed somewhere expensive — the round caps bound normal spend.

    Raised, not asserted, and NodeBudgetExceeded subclasses AssertionError
    so `graph._contain_node_budget` can turn it into the graceful-abort path
    rather than ending the process with a traceback.
    """
    if total_usd <= ceiling_usd:
        return
    raise NodeBudgetExceeded(
        f"{what} cost ${total_usd:.4f}{context} exceeds the "
        f"${ceiling_usd:.2f} {budget} — check {check} before rerunning"
    )
