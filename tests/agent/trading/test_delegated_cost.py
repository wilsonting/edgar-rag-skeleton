"""The research agent's tools spend money server-side, and until 2026-08-27
that spend reached nothing — not `docs/cost-log.jsonl`, not
`TradingState.cost_events`, and so not `check_run_guards`.

Measured on the Phase 9 battery at ~28% of a real fundamentals run, which is
what let AVGO cost ~$1.41 against a $1.10 cap without tripping it. These
tests pin the accounting path end to end, and the last one pins the property
that actually failed: the guard can see it.
"""

from __future__ import annotations

from datetime import date

import pytest

import app.agent.tools as tools
from app.agent.trading.domain.budget import CostEvent, RunBudget, total_spend
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.domain.token_usage import (
    USAGE_HEADER,
    TokenUsage,
    decode_usage_header,
    encode_usage_header,
)


class _Resp:
    def __init__(self, headers):
        self.headers = headers


def _header(*usages):
    return _Resp({USAGE_HEADER: encode_usage_header(usages)})


@pytest.fixture(autouse=True)
def _clean_accumulator():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


def test_usage_reported_by_the_api_is_accumulated_per_model():
    usage = TokenUsage(input_tokens=5158, output_tokens=500)
    tools._record_delegated_usage(_header(("deepseek-v4-flash", usage)))
    tools._record_delegated_usage(_header(("deepseek-v4-flash", usage)))

    total = tools.get_delegated_usage()["deepseek-v4-flash"]
    assert total.input_tokens == 2 * 5158
    assert total.output_tokens == 2 * 500


def test_a_response_without_the_header_is_zero_not_an_error():
    """Endpoints that spend nothing (`/corpus-status`, `/ingest`) send no
    header, and an older server sends none at all. Accounting must not be
    able to fail a run."""
    tools._record_delegated_usage(_Resp({}))
    assert tools.get_delegated_usage() == {}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"by_model": [1]}'])
def test_a_malformed_header_is_ignored_rather_than_raised(raw):
    tools._record_delegated_usage(_Resp({USAGE_HEADER: raw}))
    assert tools.get_delegated_usage() == {}


def test_the_accumulator_is_per_run():
    """`reset_run_provenance` already fences every other per-run accumulator
    in this module; usage has to be fenced by the same call or run two would
    be billed for run one."""
    tools._record_delegated_usage(_header(("m", TokenUsage(input_tokens=999))))
    assert tools.get_delegated_usage()
    tools.reset_run_provenance()
    assert tools.get_delegated_usage() == {}


def test_usage_survives_a_json_round_trip_through_the_header():
    usage = TokenUsage(
        input_tokens=1, output_tokens=2, cache_write_tokens=3, cache_read_tokens=4
    )
    assert decode_usage_header(encode_usage_header([("m", usage)])) == {"m": usage}


# ---------------------------------------------------------------------------
# Per-model pricing. Found live 2026-09-11: with /ask and /extract on
# deepseek-v4-flash under a gpt-5.6-luna agent, a bare token total priced at
# the agent's rate logged $0.0375 for ~$0.0666 of real spend.
# ---------------------------------------------------------------------------

def test_one_request_that_spends_on_two_models_reports_both():
    """/ask pays for the answer AND the decomposer, which can be different
    models. Summed before sending, the caller could never price them apart."""
    answer = TokenUsage(input_tokens=5000, output_tokens=400)
    rewrite = TokenUsage(input_tokens=300, output_tokens=60)
    decoded = decode_usage_header(encode_usage_header(
        [("deepseek-v4-flash", answer), ("gpt-5.6-luna", rewrite)]
    ))
    assert decoded == {"deepseek-v4-flash": answer, "gpt-5.6-luna": rewrite}


def test_empty_usage_is_left_out_of_the_header():
    """The decomposer's regex path makes no call; a zero entry would log a
    $0 cost event under a model that was never called."""
    decoded = decode_usage_header(encode_usage_header(
        [("deepseek-v4-flash", TokenUsage(input_tokens=10)), ("gpt-5.6-luna", TokenUsage())]
    ))
    assert list(decoded) == ["deepseek-v4-flash"]


def test_a_legacy_flat_header_is_kept_under_no_model():
    """A server started before this change still sends one bare total. It is
    accounted, not dropped, and the caller prices it at its own model as
    before."""
    legacy = TokenUsage(input_tokens=777).model_dump_json()
    tools._record_delegated_usage(_Resp({USAGE_HEADER: legacy}))
    assert tools.get_delegated_usage() == {None: TokenUsage(input_tokens=777)}


def test_the_server_endpoint_helper_emits_per_model_usage():
    from fastapi import Response

    from app.main import _report_usage

    response = Response()
    _report_usage(
        response,
        ("deepseek-v4-flash", TokenUsage(input_tokens=100)),
        ("gpt-5.6-luna", TokenUsage(output_tokens=7)),
    )
    assert decode_usage_header(response.headers[USAGE_HEADER]) == {
        "deepseek-v4-flash": TokenUsage(input_tokens=100),
        "gpt-5.6-luna": TokenUsage(output_tokens=7),
    }


@pytest.mark.anyio
async def test_fundamentals_logs_one_tool_event_per_model_at_its_own_rate(monkeypatch):
    from app.agent.researcher import UsageSummary, _compute_cost
    from app.agent.trading.infrastructure import fundamentals_port as port

    async def fake_run_agent(task, system_prompt, **_):
        return "# memo", UsageSummary()

    logged = []

    def fake_log_cost(ticker, mode, usage, model=port.AGENT_MODEL, *, run_id=None, event_id=None):
        logged.append((mode, model))
        return _compute_cost(usage, model)

    monkeypatch.setattr(port, "_USE_MOCK", False)
    monkeypatch.setattr(port, "run_agent", fake_run_agent)
    monkeypatch.setattr(port, "log_cost", fake_log_cost)
    monkeypatch.setattr(port, "_save_output", lambda *a, **k: "vault/path")
    monkeypatch.setattr(port, "AGENT_MODEL", "gpt-5.6-luna")

    tools._record_delegated_usage(_header(
        ("deepseek-v4-flash", TokenUsage(input_tokens=115_681, output_tokens=11_750)),
        ("gpt-5.6-luna", TokenUsage(input_tokens=2_000, output_tokens=300)),
    ))

    report = await port.get_fundamentals_report("ACN", date(2026, 8, 19), run_id="r1")

    by_model = {e.model: e for e in report.tool_cost_events}
    assert set(by_model) == {"deepseek-v4-flash", "gpt-5.6-luna"}
    assert ("trading-fundamentals-tools", "deepseek-v4-flash") in logged
    # Priced at DeepSeek's rate, not the agent's: 115681*0.44 + 11750*1.32.
    assert by_model["deepseek-v4-flash"].usd == pytest.approx(0.066410, abs=1e-6)


@pytest.mark.anyio
async def test_legacy_usage_is_priced_at_the_agent_model(monkeypatch):
    from app.agent.researcher import UsageSummary
    from app.agent.trading.infrastructure import fundamentals_port as port

    async def fake_run_agent(task, system_prompt, **_):
        return "# memo", UsageSummary()

    monkeypatch.setattr(port, "_USE_MOCK", False)
    monkeypatch.setattr(port, "run_agent", fake_run_agent)
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.01)
    monkeypatch.setattr(port, "_save_output", lambda *a, **k: "vault/path")

    tools._record_delegated_usage(
        _Resp({USAGE_HEADER: TokenUsage(input_tokens=10).model_dump_json()})
    )
    report = await port.get_fundamentals_report("ACN", date(2026, 8, 19), run_id="r1")
    assert [e.model for e in report.tool_cost_events] == [port.AGENT_MODEL]


def test_a_report_with_the_old_single_tool_event_keeps_its_spend():
    """The fundamentals caches and pre-change checkpoints carry one
    `tool_cost_event`. Ignored as an unknown field, a resumed run would lose
    that spend from its ledger."""
    event = CostEvent(
        event_id="fundamentals-tools:x", node="fundamentals-tools", model="m",
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
        cache_read_input_tokens=0, usd=0.25,
    )
    report = FundamentalsReport.model_validate({
        "ticker": "ACN", "summary": "s", "input_tokens": 0, "cache_write_tokens": 0,
        "cache_read_tokens": 0, "output_tokens": 0, "generated_at": "2026-09-11",
        "tool_cost_event": event.model_dump(),
    })
    assert report.tool_cost_events == [event]


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
