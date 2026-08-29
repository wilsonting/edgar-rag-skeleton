from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.trading.domain.budget import CostEvent

Sentiment = Literal["positive", "negative", "neutral"]

# How much an article is actually about the ticker under analysis. Finnhub's
# company-news feed tags broad market and sector coverage with the requested
# symbol — measured at 75% no-signal for MSFT — so the vendor's `related`
# field cannot filter. This is scored by the LLM in the same call that
# produces the summary, at no extra request.
Relevance = Literal["primary", "mentioned", "unrelated"]

# Which relevance levels the sentiment aggregate is computed over. Named
# rather than inlined so the policy is one edit, and so a test can assert
# against the same constant the node uses.
AGGREGATED_RELEVANCE: frozenset[str] = frozenset({"primary"})


class NewsItem(BaseModel):
    """One article. Every field except `summary`/`sentiment`/`relevance` is
    carried through verbatim from the vendor payload — the LLM never retypes
    them."""

    headline: str
    published_date: date        # UTC date, derived in Python from the unix ts
    source: str
    url: str
    summary: str                # LLM-generated, one line
    sentiment: Sentiment        # LLM-generated, enum-constrained
    relevance: Relevance        # LLM-generated, enum-constrained


class NewsDigest(BaseModel):
    ticker: str
    as_of_date: date            # the probe/analysis date — the upper bound
    window_start: date          # as_of_date - lookback_days
    items: list[NewsItem]

    # Provenance / audit fields. These exist so the digest can be
    # inspected without re-running the fetch.
    raw_article_count: int      # what the vendor returned
    deduped_count: int          # after dedup + window filter
    dropped_out_of_window: int  # articles the Python-side date filter rejected
    dropped_missing_date: int
    truncated_by_cap: bool      # did MAX_ARTICLES bite?
    data_source: str = "finnhub"
    cost_event: CostEvent | None = None
    # Sanitizer hits (Phase 8) across every article's headline/body at
    # ingestion — flag-not-drop, so a real article that happens to quote an
    # injection attempt is still visible in the digest, just visibly flagged.
    # Per-article flags live in NewsItem below; this is the digest-level
    # audit trail so a run's sanitizer activity is visible without walking
    # every item.
    sanitizer_flags: list[str] = Field(default_factory=list)


class SentimentSummary(BaseModel):
    """Deterministic aggregate over NewsDigest.items — no LLM.

    Counts cover only articles whose relevance is in AGGREGATED_RELEVANCE.
    `excluded_by_relevance` is not decoration: without it a consumer cannot
    tell a genuinely quiet ticker (few articles) from a noisy feed that was
    filtered down to a handful (many articles, few about this company), and
    those warrant very different confidence in `net_score`.
    """

    ticker: str
    as_of_date: date
    positive: int
    negative: int
    neutral: int
    net_score: float            # (pos - neg) / total, 0.0 when total == 0
    article_count: int          # articles the aggregate covers
    excluded_by_relevance: int  # digest items dropped before aggregating
