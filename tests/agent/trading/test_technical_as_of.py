"""Gate C / Phase 6 exit criterion 6, at the node level: `technical_node`
must produce a report whose `as_of_date` never exceeds `state["as_of_date"]`
— even when the vendor hands back bars past that date. Complements
test_price_data_port.py's lower-level tests of `_bound_to_as_of` and the
lookahead post-assert by exercising the real node body end to end, the same
way `technical_report.as_of_date = df.index[-1].date()` actually gets set.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import app.agent.trading.application.nodes as nodes


def _df(end: date, days: int) -> pd.DataFrame:
    """A clean OHLCV frame running from `end - days` up to AND PAST `end` —
    the vendor-leaks-future-bars scenario `get_price_history` has to guard
    against, exercised here through the real node rather than a mock."""
    idx = pd.date_range(end - timedelta(days=days), end + timedelta(days=10), freq="D")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=idx,
    )


@pytest.mark.anyio
async def test_technical_node_never_reports_a_date_after_as_of(monkeypatch):
    """Mocked at the yfinance SDK boundary, not at `_try_yfinance` itself —
    so the real `_try_yfinance` (and its real `_bound_to_as_of` call) runs,
    proving the silent-bounding path works end to end through the node, not
    just the defensive assert one layer up catching an unbounded mock."""
    as_of = date(2025, 3, 1)

    class _FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None):
            return _df(as_of, 400)

    import app.agent.trading.infrastructure.price_data_port as pdp

    monkeypatch.setattr(pdp.yf, "Ticker", _FakeTicker)

    async def fake_interpret(ticker, indicators):
        return "stub interpretation, no numbers.", [], [], None

    monkeypatch.setattr(nodes, "interpret_indicators", fake_interpret)
    monkeypatch.setattr(nodes, "save_technical_report", lambda report, cost_usd=None: None)

    update = await nodes.technical_node({"ticker": "ACN", "as_of_date": as_of})

    report = update["technical_report"]
    assert report.as_of_date <= as_of


@pytest.mark.anyio
async def test_technical_node_refuses_to_run_without_as_of():
    with pytest.raises(ValueError, match="lookahead"):
        await nodes.technical_node({"ticker": "ACN"})
