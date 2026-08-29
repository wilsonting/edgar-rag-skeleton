"""Two cost fixes from the Phase 9 audit, with honest scope.

`calculate` memoisation does NOT save API money — it is pure Python, and the
expensive part of a duplicate (the agent turn, a full context round-trip) is
already spent by the time `_dispatch` runs. What it buys is a signal back to
the agent that it is repeating itself. Measured cause: NFLX ran
"10149273 - 688220" three times in one run.

`check_latest_filings` narrowing DOES save directly: NFLX's call returned 44
filings of which 38 were 8-Ks, ~1,700 tokens that then rode in context for
the remaining ~43 turns.
"""

from __future__ import annotations

import pytest

import app.agent.tools as tools
from app.infrastructure.edgar.client import (
    DOMESTIC_FORM_TYPES,
    FOREIGN_PRIVATE_ISSUER_FORM_TYPES,
    periodic_forms,
)


@pytest.fixture(autouse=True)
def _fresh():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


# --- calculate memoisation -------------------------------------------------

def test_whitespace_is_the_only_thing_normalized():
    assert tools._normalize_expression("10149273 - 688220") == "10149273-688220"
    assert tools._normalize_expression(" a  +\tb ") == "a+b"


def test_scale_variants_are_kept_apart():
    """These reduce to the same ratio, and are still different expressions.
    A cache is not the place to assert two derivations are equivalent."""
    a = tools._normalize_expression("(45183.036 - 39001.0) / 39001.0 * 100")
    b = tools._normalize_expression("(45183036 - 39001000) / 39001000 * 100")
    assert a != b


def _declared(expression: str, *values: float) -> dict:
    """A `calculate` call whose figures are declared and provenanced, which
    is what the validator requires before anything is computed at all."""
    for value in values:
        tools.record_tool_output(f"FY2025 figure: {value:.0f}")
    return {
        "expression": expression,
        "inputs": [
            {
                "value": value,
                "label": f"figure {value:.0f}",
                "fiscal_period": "FY2025",
                "source": "10-K",
                "unit": "thousands",
            }
            for value in values
        ],
    }


@pytest.mark.anyio
async def test_a_repeated_expression_returns_the_same_value_and_says_so():
    args = _declared("10149273 - 688220", 10149273, 688220)

    first = await tools._dispatch("calculate", args)
    assert "Rejected" not in first
    second = await tools._dispatch("calculate", dict(args))

    assert second.startswith(first)
    assert "already computed earlier this run" in second


@pytest.mark.anyio
async def test_the_repeat_note_carries_no_digits_of_its_own():
    """Tool results are the agent's provenance corpus and the containment
    guards scan it for figures that could "back" a memo claim. A note that
    introduced a number would put one there."""
    args = _declared("45183036 - 39001000", 45183036, 39001000)
    await tools._dispatch("calculate", args)
    second = await tools._dispatch("calculate", dict(args))

    note = second[second.index("["):]
    assert not any(ch.isdigit() for ch in note)


@pytest.mark.anyio
async def test_a_repeat_does_not_double_record_the_calc_result():
    """`get_calc_results` feeds the memo verifier's `computed_values`;
    recording the same figure twice inflates it for no reason."""
    args = _declared("33723470 - 19715368", 33723470, 19715368)
    first = await tools._dispatch("calculate", args)
    assert "Rejected" not in first
    after_first = list(tools.get_calc_results())
    await tools._dispatch("calculate", dict(args))

    assert tools.get_calc_results() == after_first


def test_the_cache_is_per_run():
    tools._CALC_CACHE["x"] = "1"
    tools.reset_run_provenance()
    assert tools._CALC_CACHE == {}


# --- check_latest_filings narrowing ---------------------------------------

def test_domestic_narrows_to_periodic_reports():
    assert periodic_forms(DOMESTIC_FORM_TYPES) == ["10-K", "10-Q"]


def test_foreign_private_issuers_narrow_to_20f_not_10k():
    """The narrowing must not undo the domestic/foreign detection that
    produced the family — ASML files 20-F, and a hardcoded ["10-K","10-Q"]
    would silently return nothing for it."""
    assert periodic_forms(FOREIGN_PRIVATE_ISSUER_FORM_TYPES) == ["20-F"]


def test_an_explicitly_named_family_is_left_alone():
    """A caller that named its forms has already said what it wants."""
    assert periodic_forms(["8-K"]) == ["8-K"]


def test_the_agent_can_ask_for_event_filings_when_it_needs_them():
    schema = next(
        t for t in tools.TOOLS if t["name"] == "check_latest_filings"
    )["input_schema"]
    assert "form_types" in schema["properties"]
