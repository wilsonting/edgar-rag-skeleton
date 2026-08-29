"""The research agent's tools spend money server-side, and until 2026-08-27
that spend reached nothing — not `docs/cost-log.jsonl`, not
`TradingState.cost_events`, and so not `check_run_guards`.

Measured on the Phase 9 battery at ~28% of a real fundamentals run, which is
what let AVGO cost ~$1.41 against a $1.10 cap without tripping it. These
tests pin the accounting path end to end, and the last one pins the property
that actually failed: the guard can see it.
"""

from __future__ import annotations

import json

import pytest

import app.agent.tools as tools
from app.agent.trading.domain.budget import CostEvent, RunBudget, total_spend
from app.domain.token_usage import USAGE_HEADER, TokenUsage


class _Resp:
    def __init__(self, headers):
        self.headers = headers


@pytest.fixture(autouse=True)
def _clean_accumulator():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


def test_usage_reported_by_the_api_is_accumulated():
    usage = TokenUsage(input_tokens=5158, output_tokens=500)
    tools._record_delegated_usage(_Resp({USAGE_HEADER: usage.model_dump_json()}))
    tools._record_delegated_usage(_Resp({USAGE_HEADER: usage.model_dump_json()}))

    total = tools.get_delegated_usage()
    assert total.input_tokens == 2 * 5158
    assert total.output_tokens == 2 * 500


def test_a_response_without_the_header_is_zero_not_an_error():
    """Endpoints that spend nothing (`/corpus-status`, `/ingest`) send no
    header, and an older server sends none at all. Accounting must not be
    able to fail a run."""
    tools._record_delegated_usage(_Resp({}))
    assert tools.get_delegated_usage().is_empty


def test_a_malformed_header_is_ignored_rather_than_raised():
    tools._record_delegated_usage(_Resp({USAGE_HEADER: "not json"}))
    assert tools.get_delegated_usage().is_empty


def test_the_accumulator_is_per_run():
    """`reset_run_provenance` already fences every other per-run accumulator
    in this module; usage has to be fenced by the same call or run two would
    be billed for run one."""
    tools._record_delegated_usage(
        _Resp({USAGE_HEADER: TokenUsage(input_tokens=999).model_dump_json()})
    )
    assert not tools.get_delegated_usage().is_empty
    tools.reset_run_provenance()
    assert tools.get_delegated_usage().is_empty


def test_usage_survives_a_json_round_trip_through_the_header():
    usage = TokenUsage(
        input_tokens=1, output_tokens=2, cache_write_tokens=3, cache_read_tokens=4
    )
    assert TokenUsage.model_validate(json.loads(usage.model_dump_json())) == usage


# ---------------------------------------------------------------------------
# The property that actually broke: the run-level guard has to SEE it.
# ---------------------------------------------------------------------------

def test_delegated_spend_counts_against_the_run_budget():
    """AVGO's real cost was ~$1.41 against a $1.10 cap and the guard never
    fired, because the delegated share never entered `cost_events`. With
    both events in the ledger, `total_spend` reflects the real number."""
    own = CostEvent(
        event_id="fundamentals:aaa", node="fundamentals", model="m",
        input_tokens=918, output_tokens=22063, cache_creation_input_tokens=94795,
        cache_read_input_tokens=1408914, usd=0.3706,
    )
    delegated = CostEvent(
        event_id="fundamentals-tools:bbb", node="fundamentals-tools", model="m",
        input_tokens=206320, output_tokens=20000, cache_creation_input_tokens=0,
        cache_read_input_tokens=0, usd=0.3759,
    )

    assert total_spend([own]) == pytest.approx(0.3706)
    assert total_spend([own, delegated]) == pytest.approx(0.7465)

    budget = RunBudget(max_usd=0.60, deadline_utc="2030-01-01T00:00:00Z")
    assert total_spend([own]) < budget.max_usd          # the old blind spot
    assert total_spend([own, delegated]) > budget.max_usd  # now visible


# ---------------------------------------------------------------------------
# `from_response` is the one link that could not be verified against a live
# call: the account ran out of credits mid-session, so every real /ask
# returned 400 before producing a usage object. The header path itself WAS
# verified live end to end (FastAPI emits it, the client parses it) using an
# empty-retrieval question, which returns before any LLM call and correctly
# reports zeros. These pin the non-zero case.
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_reads_a_real_anthropic_usage_object():
    usage = TokenUsage.from_response(
        _Usage(input_tokens=5158, output_tokens=487,
               cache_creation_input_tokens=0, cache_read_input_tokens=0)
    )
    assert usage.input_tokens == 5158
    assert usage.output_tokens == 487
    assert not usage.is_empty


def test_missing_cache_fields_read_as_zero_not_as_an_error():
    """Responses from calls that set no `cache_control` — which is every
    call this covers today — carry no cache attributes at all."""
    usage = TokenUsage.from_response(_Usage(input_tokens=10, output_tokens=2))
    assert usage.cache_write_tokens == 0
    assert usage.cache_read_tokens == 0


def test_a_none_valued_token_field_reads_as_zero():
    """The SDK reports None rather than 0 for cache fields on some
    responses; `or 0` in from_response covers it, and this pins that."""
    usage = TokenUsage.from_response(
        _Usage(input_tokens=10, output_tokens=2,
               cache_creation_input_tokens=None, cache_read_input_tokens=None)
    )
    assert usage.cache_write_tokens == 0
