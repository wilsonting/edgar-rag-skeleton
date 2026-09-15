"""The one implementation of "call this tool, retry once, account for both".

The debate, risk and synthesis ports each carried their own copy of the same
six helpers. `_accumulate` was byte-identical in all three. `_extract`,
`_tool_block`, `_retry_messages`, `_CORRECTION` and the schema-retry loop
differed only in which payload class and tool name they named — and the
synthesis versions had already been generalised to take both as arguments.
`_assert_within_budget` existed four times, counting the news digest.

Three implementations of one contract is three places to keep a fix and
three chances to keep two of them. These tests pin the contract once.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from app.agent.researcher import UsageSummary
from app.agent.trading.domain.budget import NodeBudgetExceeded
from app.agent.trading.infrastructure.structured_call import (
    accumulate,
    assert_within_budget,
    call_with_schema_retry,
    extract,
    retry_messages,
    tool_block,
)


class _Payload(BaseModel):
    side: str
    claim: str


def _usage(**over):
    return SimpleNamespace(**{
        "input_tokens": 10, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0, "output_tokens": 5, **over,
    })


def _tool_use(payload: dict, tool_id: str = "t1"):
    return SimpleNamespace(type="tool_use", id=tool_id, name="submit", input=payload)


def _response(blocks, stop_reason="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=_usage())


GOOD = {"side": "bull", "claim": "margins expanded"}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_a_valid_tool_call_becomes_the_payload():
    assert extract(_response([_tool_use(GOOD)]), _Payload, "submit").side == "bull"


def test_no_tool_call_is_a_validation_error_not_an_attributeerror():
    """Same class of failure as a malformed one, and it gets the same single
    retry — a response with no tool block used to be indistinguishable from
    a crash."""
    with pytest.raises(ValidationError):
        extract(_response([SimpleNamespace(type="text", text="sorry")]), _Payload, "submit")


def test_tool_block_finds_the_first_tool_use_among_others():
    blocks = [SimpleNamespace(type="text", text="thinking"), _tool_use(GOOD)]
    assert tool_block(_response(blocks)).input == GOOD
    assert tool_block(_response([SimpleNamespace(type="text", text="x")])) is None


# ---------------------------------------------------------------------------
# The correction turn
# ---------------------------------------------------------------------------

def test_the_correction_answers_every_tool_use_with_a_tool_result():
    """A tool_use block MUST be answered by a tool_result in the next
    message — a plain user turn after one is a 400. Answering only the first
    of several is the same 400 in a different disguise."""
    response = _response([_tool_use(GOOD, "a"), _tool_use(GOOD, "b")])
    turns = retry_messages([{"role": "user", "content": "go"}], response, ValueError("bad"), "submit")

    results = turns[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert all(r["is_error"] for r in results)
    assert "submit" in results[0]["content"]


def test_a_response_with_no_tool_call_gets_plain_user_text():
    """There is no id to attach a tool_result to."""
    response = _response([SimpleNamespace(type="text", text="sorry")])
    turns = retry_messages([{"role": "user", "content": "go"}], response, ValueError("bad"), "submit")
    assert isinstance(turns[-1]["content"], str)
    assert "did not validate" in turns[-1]["content"]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_clean_call_makes_one_request():
    calls = []

    async def submit(messages):
        calls.append(messages)
        return _response([_tool_use(GOOD)])

    usage = UsageSummary()
    payload = await call_with_schema_retry(
        submit, payload_cls=_Payload, tool_name="submit",
        messages=[{"role": "user", "content": "go"}], usage=usage, label="[t]",
    )
    assert payload.claim == "margins expanded"
    assert len(calls) == 1
    assert usage.output_tokens == 5


@pytest.mark.anyio
async def test_a_schema_violation_gets_exactly_one_retry():
    responses = [_response([_tool_use({"side": "bull"})]), _response([_tool_use(GOOD)])]

    async def submit(messages):
        return responses.pop(0)

    usage = UsageSummary()
    payload = await call_with_schema_retry(
        submit, payload_cls=_Payload, tool_name="submit",
        messages=[{"role": "user", "content": "go"}], usage=usage, label="[t]",
    )
    assert payload.side == "bull"
    assert responses == []


@pytest.mark.anyio
async def test_both_attempts_are_billed_even_when_the_first_failed():
    """The first call was paid for whether or not it validated."""
    responses = [_response([_tool_use({"side": "bull"})]), _response([_tool_use(GOOD)])]

    async def submit(messages):
        return responses.pop(0)

    usage = UsageSummary()
    await call_with_schema_retry(
        submit, payload_cls=_Payload, tool_name="submit",
        messages=[{"role": "user", "content": "go"}], usage=usage, label="[t]",
    )
    assert usage.input_tokens == 20 and usage.output_tokens == 10


@pytest.mark.anyio
async def test_a_second_failure_raises_out_rather_than_looping():
    """Retries inside a node are invisible to the checkpointer, so an
    unbounded retry is a runaway no round cap can see."""
    attempts = 0

    async def submit(messages):
        nonlocal attempts
        attempts += 1
        return _response([_tool_use({"side": "bull"})])

    with pytest.raises(ValidationError):
        await call_with_schema_retry(
            submit, payload_cls=_Payload, tool_name="submit",
            messages=[{"role": "user", "content": "go"}],
            usage=UsageSummary(), label="[t]",
        )
    assert attempts == 2


def test_accumulate_treats_a_missing_cache_field_as_zero():
    usage = UsageSummary()
    accumulate(usage, _usage(cache_creation_input_tokens=None, cache_read_input_tokens=None))
    assert usage.cache_write_tokens == 0 and usage.cache_read_tokens == 0


# ---------------------------------------------------------------------------
# The spend ceiling, in the four wordings the ports had
# ---------------------------------------------------------------------------

def test_under_the_ceiling_is_silent():
    assert_within_budget(0.10, 0.35, what="debate", budget="b", check="c") is None


@pytest.mark.parametrize("kwargs,expected", [
    (
        dict(what="debate", context=" for ACN",
             budget="per-debate budget after 4 turn(s)",
             check="DEBATE_MODEL routing and the evidence pack size"),
        "debate cost $0.4000 for ACN exceeds the $0.35 per-debate budget after "
        "4 turn(s) — check DEBATE_MODEL routing and the evidence pack size "
        "before rerunning",
    ),
    (
        dict(what="risk panel", context=" for ACN",
             budget="per-panel budget after 4 turn(s)",
             check="RISK_MODEL routing and the evidence pack size"),
        "risk panel cost $0.4000 for ACN exceeds the $0.35 per-panel budget "
        "after 4 turn(s) — check RISK_MODEL routing and the evidence pack "
        "size before rerunning",
    ),
    (
        dict(what="news digest", budget="per-run budget",
             check="TRADING_NEWS_DIGEST_MODEL routing"),
        "news digest cost $0.4000 exceeds the $0.35 per-run budget — check "
        "TRADING_NEWS_DIGEST_MODEL routing before rerunning",
    ),
])
def test_each_ports_message_is_preserved(kwargs, expected):
    """The wording says which knob to check, and that was the useful part of
    having four copies. It survives."""
    with pytest.raises(NodeBudgetExceeded) as exc:
        assert_within_budget(0.40, 0.35, **kwargs)
    assert str(exc.value) == expected


def test_a_breach_is_still_an_assertionerror():
    """graph._contain_node_budget turns it into the graceful-abort path."""
    assert issubclass(NodeBudgetExceeded, AssertionError)
