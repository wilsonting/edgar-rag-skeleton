"""One pipeline run, one vault folder.

The regression worth guarding is not the path format — it is that the run's
artifacts are written at two different TIMES, through two different call
stacks. `technical` and `fundamentals` save from inside their nodes while
the graph is still executing; `sentiment`, `decision` and the debate
transcript save from the CLI after `ainvoke` returns. Anything derived per
call splits them, so this exercises both halves rather than calling
`_save_output` five times in a row.
"""

from __future__ import annotations

from datetime import date, datetime

import app.agent.researcher as researcher
from app.agent.trading.domain.debate import DebateClaim, DebateTurn, DebateTurnPayload
from app.agent.trading.domain.decision_memo import DecisionMemo, Verdict
from app.agent.trading.domain.news_digest import NewsDigest, NewsItem, SentimentSummary
from app.agent.trading.domain.technical_report import TechnicalIndicators, TechnicalReport
from app.agent.trading.infrastructure.debate_port import save_debate_transcript
from app.agent.trading.infrastructure.technical_interpreter_port import (
    save_technical_report,
)
from app.agent.trading.interface.cli import _save_vault_artifacts

AS_OF = date(2026, 8, 22)
STARTED = datetime(2026, 8, 22, 7, 1, 53)


def _technical() -> TechnicalReport:
    return TechnicalReport(
        ticker="ACN",
        as_of_date=AS_OF,
        data_source="yfinance",
        bars_used=252,
        indicators=TechnicalIndicators(last_close=327.55, rsi_14=41.2033),
        interpretation="Momentum is soft.",
    )


def _digest() -> NewsDigest:
    return NewsDigest(
        ticker="ACN",
        as_of_date=AS_OF,
        window_start=date(2026, 8, 8),
        items=[
            NewsItem(
                headline="Acme beats estimates",
                published_date=AS_OF,
                source="wire",
                url="https://example.com/1",
                summary="Results above expectations.",
                sentiment="positive",
                relevance="primary",
            )
        ],
        raw_article_count=1,
        deduped_count=1,
        dropped_out_of_window=0,
        dropped_missing_date=0,
        truncated_by_cap=False,
    )


def _turn() -> DebateTurn:
    return DebateTurn(
        turn_index=0,
        round_num=1,
        side="bull",
        payload=DebateTurnPayload(
            stance="hold",
            argument="Margins hold up.",
            claims=[
                DebateClaim(claim_id="margin-hold", text="Stable.", evidence_ref="none")
            ],
        ),
        estimated_cost_usd=0.03,
    )


def _memo() -> DecisionMemo:
    return DecisionMemo(
        ticker="ACN",
        bull_case="STUB",
        bear_case="STUB",
        risk_debate_summary="STUB",
        technical_signal="Momentum is soft.",
        reasoning="STUB",
        watch_items=[],
        verdict=Verdict.HOLD,
        confidence=0.0,
        data_as_of_date=AS_OF,
    )


def _result() -> dict:
    return {
        "ticker": "ACN",
        "news_digest": _digest(),
        "sentiment_summary": SentimentSummary(
            ticker="ACN",
            as_of_date=AS_OF,
            positive=1,
            negative=0,
            neutral=0,
            net_score=1.0,
            article_count=1,
            excluded_by_relevance=0,
        ),
        "debate_turns": [_turn()],
        "debate_terminated_by": "round_cap",
        "decision_memo": _memo(),
    }


def test_mid_run_and_end_of_run_artifacts_share_one_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(STARTED):
        # written from inside technical_node, mid-graph
        technical = save_technical_report(_technical(), cost_usd=0.001)
        # written from the CLI once ainvoke has returned
        saved = _save_vault_artifacts(_result(), "the captured run log")

    run_dir = tmp_path / "ACN" / "20260822" / "2026-0822-070153"
    assert technical.parent == run_dir
    assert {p.parent for p in saved} == {run_dir}
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "ACN-debate.md",
        "ACN-decision.md",
        "ACN-sentiment-provenance.md",
        "ACN-sentiment.md",
        "ACN-technical.md",
    ]


def test_the_debate_transcript_joins_its_run(tmp_path, monkeypatch):
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(STARTED):
        transcript = save_debate_transcript("acn", [_turn()], "round_cap")

    assert transcript == (
        tmp_path / "ACN" / "20260822" / "2026-0822-070153" / "ACN-debate.md"
    )
    assert "Bull/Bear Debate" in transcript.read_text()


def test_a_partial_run_writes_only_what_it_produced(tmp_path, monkeypatch):
    """`--only technical` has no digest and no debate. The folder should hold
    what the run actually made, and the run log rides with the memo instead
    of the sentiment report."""
    monkeypatch.setattr(researcher, "MEMO_DIR", tmp_path)

    with researcher.vault_run(STARTED):
        save_technical_report(_technical())
        _save_vault_artifacts(
            {"ticker": "ACN", "decision_memo": _memo()}, "the captured run log"
        )

    run_dir = tmp_path / "ACN" / "20260822" / "2026-0822-070153"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "ACN-decision-provenance.md",
        "ACN-decision.md",
        "ACN-technical.md",
    ]
    assert (run_dir / "ACN-decision-provenance.md").read_text() == "the captured run log"
