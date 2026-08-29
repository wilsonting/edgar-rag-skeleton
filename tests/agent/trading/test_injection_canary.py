"""Phase 8 criterion 4's mechanical half: the sanitizer screen fires on the
canary and not on the control, with no live LLM call involved (that battery
— 1 live canary run + 1 live control run, verdict parity and payload
absence — is a manual verification step, not an automated test; it costs
real API money and needs a real model to attempt to be steered).

What THIS file proves for free: the pattern screen actually reaches the
canary fixture's payload through the real `build_digest` code path (not a
hand-rolled call to sanitize_external_text in isolation), and the payload's
presence does not change the digest's structural shape (same item count,
same successful join) — only its flags. That is what makes checks (b) and
(c) in the guide mechanical rather than a judgment call once a live run
exists to look at.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.agent.trading.infrastructure import news_digest_port
from app.agent.trading.infrastructure.news_digest_port import build_digest

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

# Unique/greppable, per the guide's own instruction — this exact string
# never appears anywhere except the canary fixture and this file's assertion.
_PAYLOAD_MARKER = "XJQ7-CANARY-PAYLOAD"


def _load(name: str) -> dict:
    article = json.loads((_FIXTURES / name).read_text())
    article["_pub_date"] = date(2026, 8, 20)
    return article


@pytest.fixture(autouse=True)
def _no_llm_call(monkeypatch):
    """The digest LLM call is stubbed to a benign summary regardless of
    input — this file tests the sanitizer's own behavior, not whether a
    real model resists the injection (that's the live battery)."""
    async def fake_summarize(client, batch, ticker):
        return (
            [
                {
                    "index": i,
                    "summary": "Company reported quarterly results.",
                    "sentiment": "neutral",
                    "relevance": "primary",
                }
                for i in range(len(batch))
            ],
            _FakeUsage(),
        )

    monkeypatch.setattr(news_digest_port, "_summarize_batch", fake_summarize)
    monkeypatch.setattr(news_digest_port, "log_cost", lambda *a, **k: 0.0)


class _FakeUsage:
    input_tokens = 100
    output_tokens = 20
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


@pytest.mark.anyio
async def test_canary_fixture_trips_the_sanitizer_flag():
    canary = _load("canary_article.json")

    items, issues, cost, cost_event, flags = await build_digest([canary], "ACME")

    assert len(items) == 1
    assert issues == []
    assert any("instruction-like" in f for f in flags), (
        "the pattern screen did not fire on the canary fixture through the "
        "real build_digest path"
    )


@pytest.mark.anyio
async def test_control_fixture_does_not_trip_the_sanitizer():
    control = _load("control_article.json")

    items, issues, cost, cost_event, flags = await build_digest([control], "ACME")

    assert len(items) == 1
    assert flags == []


@pytest.mark.anyio
async def test_canary_and_control_produce_the_same_digest_shape():
    """Same structural outcome either way — the sanitizer changes what gets
    FLAGGED, never what gets DROPPED (flag-not-assert). A shape difference
    here would mean the canary's presence altered something beyond the
    flag, which is exactly the silent-censorship failure mode this design
    is supposed to avoid."""
    canary = _load("canary_article.json")
    control = _load("control_article.json")

    canary_items, canary_issues, *_ = await build_digest([canary], "ACME")
    control_items, control_issues, *_ = await build_digest([control], "ACME")

    assert len(canary_items) == len(control_items) == 1
    assert canary_issues == control_issues == []
    assert canary_items[0].sentiment == control_items[0].sentiment
    assert canary_items[0].relevance == control_items[0].relevance


@pytest.mark.anyio
async def test_payload_marker_never_reaches_the_summary_the_model_authored():
    """The digest's `summary` field is LLM-authored (Python never copies
    vendor prose into it) — stubbed here to a fixed benign string precisely
    so this assertion is about the JOIN, not about model behavior. The
    payload marker living in the sanitized `headline`/body is expected and
    fine (flag-not-drop); it appearing in the OUTPUT `summary` would mean
    the join leaked raw article text into a field that is supposed to be
    the model's own words.
    """
    canary = _load("canary_article.json")
    items, _, *_ = await build_digest([canary], "ACME")

    assert _PAYLOAD_MARKER not in items[0].summary
