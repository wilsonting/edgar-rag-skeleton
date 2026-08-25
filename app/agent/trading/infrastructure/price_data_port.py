from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone

import finnhub
import pandas as pd
import yfinance as yf

from app.agent.trading.domain.errors import VendorError

MIN_BARS_REQUIRED = 210   # 200-day SMA + small buffer for weekends/holidays

# Trailing window fetched, ending at `as_of` rather than at "now". 400
# calendar days covers the 200-bar SMA plus weekends/holidays with margin —
# the same figure _try_finnhub already used when it fetched relative to
# datetime.now(); `as_of` replaces "now" as the anchor, the width is
# unchanged.
_LOOKBACK_DAYS = 400


async def get_price_history(ticker: str, as_of: date) -> tuple[pd.DataFrame, str, int]:
    """Returns (OHLCV DataFrame, source_name, bars_dropped_invalid), bounded
    at `as_of` — no bar dated after it is ever returned.

    Gate C (Phase 6 plan §0): `technical_node` used to fetch `period="1y"`,
    which yfinance anchors at wall-clock "now" regardless of the run's
    `--as-of`. A historical probe run got a technical report computed from
    bars the run was not supposed to be able to see yet — a lookahead leak
    identical in shape to the one `news_node`'s as_of guard and post-assert
    already close for news, just never fixed here. Confirmed live
    (AVGO, 2026-08-23, `--as-of 2026-08-20`): the technical report came back
    dated 2026-08-21, one day past the run's stated bound, and six debate
    turns then argued over that bar in detail. See trading-agent-known-gaps.md
    item 16.

    Raises VendorError if both vendors fail or return insufficient
    ON-OR-BEFORE-`as_of` history. Bars with a missing Close or Volume are
    dropped here, at the vendor boundary, so every downstream consumer sees
    the same clean frame and `bars_used` counts what was actually computed
    on. See _drop_invalid_bars.
    """
    df, source = await asyncio.to_thread(_try_yfinance, ticker, as_of)
    if df is None or len(_drop_invalid_bars(df)[0]) < MIN_BARS_REQUIRED:
        df, source = await asyncio.to_thread(_try_finnhub, ticker, as_of)

    if df is None:
        raise VendorError(f"No price data for {ticker} from yfinance or Finnhub")

    # Cleaned before the sufficiency check, not after: a frame padded out to
    # MIN_BARS_REQUIRED with unusable rows does not have enough history for a
    # 200-day SMA, and passing it through would just move the failure.
    df, dropped = _drop_invalid_bars(df)

    if len(df) < MIN_BARS_REQUIRED:
        raise VendorError(
            f"{ticker}: only {len(df)} usable bars from {source} on or before "
            f"{as_of} ({dropped} dropped as incomplete), need "
            f"{MIN_BARS_REQUIRED} for 200-day SMA (recent IPO, thin ticker, "
            f"or --as-of set too far in the past?)"
        )

    # Belt-and-braces, same posture as news_node's lookahead post-assert: a
    # vendor quirk (clock skew, a "today" bar keyed to the wrong side of
    # midnight) putting one bar past `as_of` into the frame would otherwise
    # be a silent contamination rather than a loud one.
    late = df.index[df.index.date > as_of] if len(df) else df.index
    if len(late):
        raise AssertionError(
            f"Lookahead leak: {len(late)} price bar(s) dated after {as_of} "
            f"reached technical_node — first: {late[0].date()}"
        )
    return df, source, dropped


def _drop_invalid_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with a missing Close or Volume, and report how many.

    Not defensive housekeeping — a single bad bar silently destroyed a whole
    indicator in production. yfinance returned MSFT with a NaN Close on bar 0
    (2025-08-21); MACD is built on exponential moving averages, which seed
    from the first observation and propagate NaN forward, so all 251
    subsequent rows came back NaN while SMA and RSI were unaffected (their
    trailing windows never reach bar 0). The memo reported "MACD data is
    unavailable" and no one could see why. With the bar dropped the same run
    yields MACD 20.02 / signal 23.25 / histogram -3.23 — a bearish crossover
    that had been invisible.

    Dropping rather than interpolating is deliberate: an interpolated bar is
    a number the vendor never published, and this pipeline reports figures it
    can trace. A dropped bar is a gap; an invented one is a fabrication.

    Volume is included because volume_vs_20d_avg divides by a 20-bar mean —
    a NaN there yields NaN, which is a valid float to pydantic and would
    reach the report as `NaN`, and JSON has no such literal.
    """
    required = [c for c in ("Close", "Volume") if c in df.columns]
    if not required:
        return df, 0
    clean = df.dropna(subset=required)
    return clean, len(df) - len(clean)


def _bound_to_as_of(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Drop any bar dated after `as_of`. Applied inside both vendor helpers
    so the AssertionError in get_price_history is a last-resort catch of a
    vendor anomaly, not the primary bounding mechanism."""
    return df[df.index.date <= as_of]


def _try_yfinance(ticker: str, as_of: date) -> tuple[pd.DataFrame | None, str]:
    try:
        start = as_of - timedelta(days=_LOOKBACK_DAYS)
        # yfinance's `end` is exclusive, and takes a date with no time
        # component ambiguity the way "now" would; +1 day so `as_of` itself
        # is included.
        df = yf.Ticker(ticker).history(
            start=start, end=as_of + timedelta(days=1), interval="1d"
        )
        if df is None or df.empty:
            return None, "yfinance"
        return _bound_to_as_of(df, as_of), "yfinance"
    except Exception:
        return None, "yfinance"


def _try_finnhub(ticker: str, as_of: date) -> tuple[pd.DataFrame | None, str]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return None, "finnhub"
    try:
        client = finnhub.Client(api_key=api_key)
        # End of day UTC on `as_of`, not datetime.now() — the whole point of
        # threading as_of through is that a historical probe must never ask
        # a vendor for anything past its stated bound.
        to_ts = int(
            datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc)
            .timestamp()
        )
        from_ts = to_ts - _LOOKBACK_DAYS * 24 * 60 * 60
        candles = client.stock_candles(ticker, "D", from_ts, to_ts)
        if candles.get("s") != "ok":
            return None, "finnhub"
        df = pd.DataFrame(
            {
                "Open": candles["o"],
                "High": candles["h"],
                "Low": candles["l"],
                "Close": candles["c"],
                "Volume": candles["v"],
            },
            index=pd.to_datetime(candles["t"], unit="s"),
        )
        return _bound_to_as_of(df, as_of), "finnhub"
    except Exception:
        return None, "finnhub"
