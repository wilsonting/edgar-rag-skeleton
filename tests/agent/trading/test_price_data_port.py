import asyncio
from datetime import date

import pandas as pd
import pytest

from app.agent.trading.domain.errors import VendorError
from app.agent.trading.infrastructure import price_data_port as pdp

SUFFICIENT_ROWS = pdp.MIN_BARS_REQUIRED + 5
INSUFFICIENT_ROWS = 40  # e.g. a ticker that partially IPO'd mid-year
AS_OF = date(2026, 1, 1)  # after every fixture date below, so the
# lookahead post-assert in get_price_history never fires on fixture data


def _fake_df(rows: int) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100},
        index=idx,
    )


def test_falls_back_to_finnhub_when_yfinance_hard_fails(monkeypatch):
    """yfinance raising/returning None should trigger the Finnhub attempt."""
    monkeypatch.setattr(pdp, "_try_yfinance", lambda ticker, as_of: (None, "yfinance"))
    monkeypatch.setattr(
        pdp, "_try_finnhub", lambda ticker, as_of: (_fake_df(SUFFICIENT_ROWS), "finnhub")
    )

    df, source, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert source == "finnhub"
    assert len(df) == SUFFICIENT_ROWS


def test_falls_back_to_finnhub_when_yfinance_returns_insufficient_bars(monkeypatch):
    """yfinance succeeding but returning too few bars (e.g. a recent IPO) must
    still trigger the Finnhub attempt — not silently proceed with too little
    history for a 200-day SMA."""
    monkeypatch.setattr(
        pdp, "_try_yfinance", lambda ticker, as_of: (_fake_df(INSUFFICIENT_ROWS), "yfinance")
    )
    monkeypatch.setattr(
        pdp, "_try_finnhub", lambda ticker, as_of: (_fake_df(SUFFICIENT_ROWS), "finnhub")
    )

    df, source, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert source == "finnhub"
    assert len(df) == SUFFICIENT_ROWS


def test_yfinance_success_does_not_call_finnhub(monkeypatch):
    """When yfinance already returns enough bars, Finnhub should never be
    attempted — confirms the fallback is conditional, not unconditional."""
    finnhub_called = False

    def _finnhub_spy(ticker, as_of):
        nonlocal finnhub_called
        finnhub_called = True
        return _fake_df(SUFFICIENT_ROWS), "finnhub"

    monkeypatch.setattr(
        pdp, "_try_yfinance", lambda ticker, as_of: (_fake_df(SUFFICIENT_ROWS), "yfinance")
    )
    monkeypatch.setattr(pdp, "_try_finnhub", _finnhub_spy)

    df, source, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert source == "yfinance"
    assert finnhub_called is False


def test_raises_vendor_error_when_both_vendors_fail(monkeypatch):
    monkeypatch.setattr(pdp, "_try_yfinance", lambda ticker, as_of: (None, "yfinance"))
    monkeypatch.setattr(pdp, "_try_finnhub", lambda ticker, as_of: (None, "finnhub"))

    with pytest.raises(VendorError, match="No price data for TICK"):
        asyncio.run(pdp.get_price_history("TICK", AS_OF))


def test_raises_vendor_error_when_both_vendors_return_insufficient_bars(monkeypatch):
    """Both vendors return *something*, just not enough — must still raise
    rather than silently proceed with too little history for a 200-day SMA."""
    monkeypatch.setattr(
        pdp, "_try_yfinance", lambda ticker, as_of: (_fake_df(INSUFFICIENT_ROWS), "yfinance")
    )
    monkeypatch.setattr(
        pdp, "_try_finnhub", lambda ticker, as_of: (_fake_df(INSUFFICIENT_ROWS), "finnhub")
    )

    with pytest.raises(VendorError, match="only 40 usable bars"):
        asyncio.run(pdp.get_price_history("TICK", AS_OF))


# ---------------------------------------------------------------------------
# Incomplete bars — regression from a live MSFT run (2026-08-23)
# ---------------------------------------------------------------------------

def _df_with_nan_close_at(rows: int, pos: int) -> pd.DataFrame:
    df = _fake_df(rows)
    df.iloc[pos, df.columns.get_loc("Close")] = float("nan")
    return df


def test_bar_with_missing_close_is_dropped_and_counted(monkeypatch):
    """yfinance returned MSFT with a NaN Close on bar 0. MACD seeds its EWMs
    from the first observation and propagates NaN forward, so that one bar
    voided all 251 later rows while SMA and RSI were untouched — the memo
    said "MACD data is unavailable" with nothing to explain why."""
    monkeypatch.setattr(
        pdp,
        "_try_yfinance",
        lambda ticker, as_of: (_df_with_nan_close_at(SUFFICIENT_ROWS, 0), "yfinance"),
    )

    df, source, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert dropped == 1
    assert len(df) == SUFFICIENT_ROWS - 1
    assert df["Close"].isna().sum() == 0
    assert source == "yfinance"


def test_bar_with_missing_volume_is_dropped_too(monkeypatch):
    """volume_vs_20d_avg divides by a 20-bar mean; a NaN there produces NaN,
    which pydantic accepts as a float and JSON cannot represent."""
    df_in = _fake_df(SUFFICIENT_ROWS)
    df_in.iloc[-1, df_in.columns.get_loc("Volume")] = float("nan")
    monkeypatch.setattr(pdp, "_try_yfinance", lambda ticker, as_of: (df_in, "yfinance"))

    df, _, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert dropped == 1
    assert df["Volume"].isna().sum() == 0


def test_clean_history_reports_zero_dropped(monkeypatch):
    monkeypatch.setattr(
        pdp, "_try_yfinance", lambda ticker, as_of: (_fake_df(SUFFICIENT_ROWS), "yfinance")
    )

    df, _, dropped = asyncio.run(pdp.get_price_history("TICK", AS_OF))

    assert dropped == 0
    assert len(df) == SUFFICIENT_ROWS


def test_sufficiency_is_judged_on_usable_bars_not_returned_rows(monkeypatch):
    """A frame padded to the minimum with unusable rows does not have the
    history a 200-day SMA needs; accepting it would move the failure
    downstream instead of reporting it here."""
    padded = _fake_df(pdp.MIN_BARS_REQUIRED)
    for pos in range(5):
        padded.iloc[pos, padded.columns.get_loc("Close")] = float("nan")
    monkeypatch.setattr(pdp, "_try_yfinance", lambda ticker, as_of: (padded, "yfinance"))
    monkeypatch.setattr(pdp, "_try_finnhub", lambda ticker, as_of: (padded, "finnhub"))

    with pytest.raises(VendorError, match="5 dropped as incomplete"):
        asyncio.run(pdp.get_price_history("TICK", AS_OF))


def test_macd_survives_once_the_bad_bar_is_gone():
    """The point of the drop: the indicator that the NaN destroyed comes back.
    Runs the real computation, so it fails if EWM propagation returns."""
    from app.agent.trading.application.technical_indicators import compute_indicators

    rows = SUFFICIENT_ROWS
    dirty = _df_with_nan_close_at(rows, 0)
    # a trending series, so MACD is a real number rather than a flat zero
    dirty["Close"] = [float("nan")] + [100.0 + i * 0.5 for i in range(rows - 1)]

    assert compute_indicators(dirty).macd is None, "fixture no longer reproduces the bug"

    cleaned, dropped = pdp._drop_invalid_bars(dirty)

    assert dropped == 1
    assert compute_indicators(cleaned).macd is not None


# ---------------------------------------------------------------------------
# as_of bounding — Gate C, trading-agent-known-gaps.md item 16
# ---------------------------------------------------------------------------

def test_bars_after_as_of_are_dropped_even_if_a_vendor_returns_them():
    """_bound_to_as_of runs inside the (unmocked, real) vendor helpers — this
    exercises that filter directly rather than through the network."""
    df = _fake_df(SUFFICIENT_ROWS)  # 2025-01-01 .. late July 2025
    as_of = date(2025, 3, 1)

    bounded = pdp._bound_to_as_of(df, as_of)

    assert len(bounded) > 0
    assert (bounded.index.date <= as_of).all()
    assert len(bounded) < len(df)


def test_get_price_history_raises_if_a_vendor_leaks_a_future_bar(monkeypatch):
    """Defense in depth: even though _bound_to_as_of runs inside the real
    vendor helpers, a mocked helper that skips it (as these tests' lambdas
    do) must still be caught by the post-fetch assert in get_price_history —
    the same posture as news_node's lookahead post-assert."""
    monkeypatch.setattr(
        pdp, "_try_yfinance", lambda ticker, as_of: (_fake_df(SUFFICIENT_ROWS), "yfinance")
    )

    as_of = date(2025, 3, 1)  # earlier than _fake_df's last bar

    with pytest.raises(AssertionError, match="Lookahead leak"):
        asyncio.run(pdp.get_price_history("TICK", as_of))
