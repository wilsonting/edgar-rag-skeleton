"""The fundamentals leg is bounded at the run's analysis date.

Every other source in the pipeline enforces `as_of_date`: prices and news
both refuse to run without it, and the synthesizer refuses to date a memo by
the wall clock. The fundamentals leg — the one whose output carries the most
weight in the memo — never received the date at all. `get_fundamentals_report`
had no `as_of` parameter; it called `date.today()` for the task prompt and
for `generated_at`. So a `--as-of 2026-03-01` run bounded its prices and news
at March, let its analyst read whatever had been filed since, and said
nothing about it in the memo.

The bound is enforced in the TOOLS, not in the prompt. Wording tells the
model what to do; `filed_before` on every filing-reading call is what makes
it so on the turns the model is not thinking about it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import app.agent.tools as tools
import app.agent.trading.application.nodes as nodes
from app.agent.trading.domain.fundamentals_report import FundamentalsReport
from app.agent.trading.infrastructure import fundamentals_port as port

AS_OF = date(2026, 3, 1)


@pytest.fixture(autouse=True)
def _fresh():
    tools.reset_run_provenance()
    yield
    tools.reset_run_provenance()


# ---------------------------------------------------------------------------
# The node will not run unbounded
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_the_node_refuses_to_run_without_an_analysis_date():
    with pytest.raises(ValueError) as exc:
        await nodes.fundamentals_node({"ticker": "ACN"})
    assert "lookahead" in str(exc.value)


@pytest.mark.anyio
async def test_the_node_passes_the_date_to_the_port(monkeypatch):
    seen = {}

    async def fake(ticker, as_of, **kw):
        seen["ticker"], seen["as_of"] = ticker, as_of
        return None

    monkeypatch.setattr(nodes, "get_fundamentals_report", fake)
    await nodes.fundamentals_node({"ticker": "ACN", "as_of_date": AS_OF})
    assert seen == {"ticker": "ACN", "as_of": AS_OF}


# ---------------------------------------------------------------------------
# The tools carry the bound
# ---------------------------------------------------------------------------

class _Resp:
    status_code = 200
    headers: dict = {}
    text = "{}"

    def json(self):
        return {"answer": "a", "chunks": []}


def _capture(monkeypatch):
    sent = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            sent["url"], sent["json"] = url, json or {}
            return _Resp()

        async def get(self, url, params=None, **kw):
            sent["url"], sent["params"] = url, params or {}
            return _Resp()

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(tools, "USE_STUBS", False)
    return sent


@pytest.mark.anyio
async def test_ask_edgar_bounds_retrieval_at_the_analysis_date(monkeypatch):
    sent = _capture(monkeypatch)
    tools.reset_run_provenance(AS_OF)

    await tools._dispatch("ask_edgar", {"question": "q", "tickers": ["ACN"]})

    assert sent["json"]["filed_before"] == "2026-03-01"


@pytest.mark.anyio
async def test_check_latest_filings_does_not_reveal_later_filings(monkeypatch):
    """Knowing a 10-K exists is lookahead before a word of it is read."""
    sent = _capture(monkeypatch)
    tools.reset_run_provenance(AS_OF)

    await tools._dispatch("check_latest_filings", {"ticker": "ACN"})

    assert sent["json"]["filed_before"] == "2026-03-01"


@pytest.mark.anyio
async def test_extract_metrics_window_is_clamped_to_the_bound(monkeypatch):
    """The model picks this window; the run's bound caps the top of it."""
    sent = _capture(monkeypatch)
    tools.reset_run_provenance(AS_OF)

    await tools._dispatch("extract_metrics", {
        "ticker": "ACN", "fiscal_period": "FY2026", "filing_type": "10-K",
        "filed_date": "2026-01-15", "filed_before": "2026-12-31",
    })

    assert sent["json"]["filed_before"] == "2026-03-01"


@pytest.mark.anyio
async def test_a_narrower_window_the_model_chose_is_left_alone(monkeypatch):
    sent = _capture(monkeypatch)
    tools.reset_run_provenance(AS_OF)

    await tools._dispatch("extract_metrics", {
        "ticker": "ACN", "fiscal_period": "FY2026", "filing_type": "10-K",
        "filed_date": "2026-01-15", "filed_before": "2026-01-31",
    })

    assert sent["json"]["filed_before"] == "2026-01-31"


@pytest.mark.anyio
async def test_an_unbounded_run_sends_no_bound(monkeypatch):
    """The standalone research CLI and /news-assess answer about now, and
    their behaviour is unchanged."""
    sent = _capture(monkeypatch)
    tools.reset_run_provenance()      # no as_of

    await tools._dispatch("ask_edgar", {"question": "q", "tickers": ["ACN"]})

    assert "filed_before" not in sent["json"]


def test_the_bound_is_not_something_the_model_can_set():
    """A bound the agent has to remember to apply is not a bound.

    `extract_metrics` is the one tool that takes a date window, and that
    window is the model's own extraction choice rather than the run's
    bound — it is clamped, not trusted (see the two tests above). Nothing
    else offers a date argument at all.
    """
    by_name = {t["name"]: t["input_schema"]["properties"] for t in tools.TOOLS}
    for name in ("ask_edgar", "check_latest_filings", "check_corpus", "ingest_ticker"):
        assert "filed_before" not in by_name[name], name
        assert "filed_after" not in by_name[name], name


# ---------------------------------------------------------------------------
# The port dates the report by the analysis date, and caches by it
# ---------------------------------------------------------------------------

def test_the_cache_is_keyed_on_the_date_as_well_as_the_ticker():
    """On ticker alone, MOCK_FUNDAMENTALS paired an August memo with a
    March run."""
    assert port._cache_path("ACN", AS_OF) != port._cache_path("ACN", date(2026, 8, 1))
    assert "2026-03-01" in port._cache_path("ACN", AS_OF).name


@pytest.mark.anyio
async def test_the_report_is_dated_by_the_analysis_date_not_the_wall_clock(monkeypatch):
    captured = {}

    async def fake_run_agent(task, system_prompt, **kw):
        from app.agent.researcher import UsageSummary
        captured["task"], captured["as_of"] = task, kw.get("as_of")
        return "# memo", UsageSummary()

    monkeypatch.setattr(port, "_USE_MOCK", False)
    monkeypatch.setattr(port, "run_agent", fake_run_agent)
    monkeypatch.setattr(port, "log_cost", lambda *a, **k: 0.01)
    monkeypatch.setattr(port, "_save_output", lambda *a, **k: "vault/path")
    monkeypatch.setattr(port, "get_delegated_usage", lambda: {})
    monkeypatch.setattr(port, "_CACHE_DIR", __import__("pathlib").Path("/tmp/_fundamentals_test"))

    report = await port.get_fundamentals_report("ACN", AS_OF, run_id="r1")

    assert report.generated_at == AS_OF != date.today()
    assert "2026-03-01" in captured["task"]
    assert captured["as_of"] == AS_OF


# ---------------------------------------------------------------------------
# The memo says the run was historical
# ---------------------------------------------------------------------------

def _report() -> FundamentalsReport:
    return FundamentalsReport(
        ticker="ACN", summary="# memo", input_tokens=0, cache_write_tokens=0,
        cache_read_tokens=0, output_tokens=0, generated_at=AS_OF,
    )


def test_a_historical_run_says_the_models_own_priors_are_not_bounded():
    gaps = nodes._fundamentals_caveats(
        {"as_of_date": AS_OF, "fundamentals_report": _report()}
    )
    assert len(gaps) == 1
    assert "2026-03-01" in gaps[0] and "prior knowledge is not bounded" in gaps[0].replace("\n", " ")


def test_a_run_dated_today_has_no_after_to_leak():
    gaps = nodes._fundamentals_caveats(
        {"as_of_date": date.today(), "fundamentals_report": _report()}
    )
    assert gaps == []


def test_a_run_whose_fundamentals_leg_did_not_run_says_nothing_about_it():
    """That absence is already reported by the analyst-did-not-run gap."""
    assert nodes._fundamentals_caveats({"as_of_date": AS_OF - timedelta(days=1)}) == []


# ---------------------------------------------------------------------------
# check_corpus: seeing that a filing exists is lookahead too
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_check_corpus_is_bounded_at_the_analysis_date(monkeypatch):
    """Found live. A FIG run at --as-of 2026-03-01 could not READ the later
    filings — ask_edgar and latest-filings were bounded — but check_corpus
    was not, so the agent enumerated them and wrote into its own memo:

      "The corpus also contains filings dated after the analysis cutoff,
       including Form 10-Q filed 2026-05-14 and Form 10-Q filed 2026-08-05"

    Knowing a filing exists, and when, is information from after the cutoff.
    """
    sent = _capture(monkeypatch)
    tools.reset_run_provenance(AS_OF)

    await tools._dispatch("check_corpus", {"ticker": "FIG"})

    assert sent["params"]["filed_before"] == "2026-03-01"


@pytest.mark.anyio
async def test_an_unbounded_run_still_sees_the_whole_corpus(monkeypatch):
    sent = _capture(monkeypatch)
    tools.reset_run_provenance()

    await tools._dispatch("check_corpus", {"ticker": "FIG"})

    assert "filed_before" not in sent["params"]
