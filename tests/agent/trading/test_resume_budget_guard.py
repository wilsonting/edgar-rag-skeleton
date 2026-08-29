"""A resume inherits the checkpoint's budget and deadline, never the flags on
the command line. That rule is deliberate — it keeps a breach test
deterministic — and it had a hole: nothing said so, and nothing checked
whether the inherited deadline had already passed.

Live cost of the hole (MSFT, 2026-08-28): a thread whose first attempt died
~17 hours earlier was resumed with --max-usd 1.40. It silently used the
checkpointed 1.10, ran the whole fundamentals stage, then aborted
`deadline_exceeded` on the first guard check after it — $0.4069, no memo.
The guards can only fire between nodes, so on this graph the cheapest place
they can catch a doomed run is already past the expensive part.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.trading.domain.budget import RunBudget
from app.agent.trading.interface.cli import _describe_stale_budget, _humanize


def _budget(max_usd=1.10, **offset):
    return RunBudget(
        max_usd=max_usd,
        deadline_utc=datetime.now(timezone.utc) + timedelta(**offset),
    )


def test_a_live_deadline_with_matching_budget_proceeds():
    assert _describe_stale_budget({"budget": _budget(minutes=20)}, 1.10, 1800.0) is None


def test_an_expired_deadline_refuses():
    msg = _describe_stale_budget({"budget": _budget(hours=-17)}, 1.10, 1800.0)
    assert msg is not None
    assert "REFUSING TO RESUME" in msg


def test_the_refusal_explains_the_cost_rather_than_just_saying_no():
    """The whole point is that the failure was expensive and non-obvious.
    A bare 'cannot resume' would send the reader to the source to find out
    why the deadline did not simply restart."""
    msg = _describe_stale_budget({"budget": _budget(hours=-2)}, 1.10, 1800.0)
    assert "absolute instant" in msg
    assert "no memo" in msg
    assert "--thread-id" in msg  # tells them what to do instead


def test_an_ignored_max_usd_flag_is_called_out():
    """Silently ignoring a flag the operator just typed is how $0.4069 got
    spent against a cap the operator thought they had raised."""
    msg = _describe_stale_budget({"budget": _budget(minutes=20)}, 1.40, 1800.0)
    assert msg is not None
    assert "IGNORED" in msg
    assert "1.40" in msg and "1.10" in msg


def test_a_matching_max_usd_is_not_called_out():
    assert _describe_stale_budget({"budget": _budget(1.40, minutes=20)}, 1.40, 1800.0) is None


def test_a_checkpoint_without_a_budget_proceeds():
    """Pre-Phase-8 threads carry no budget at all; they must stay resumable."""
    assert _describe_stale_budget({}, 1.10, 1800.0) is None


def test_humanize_reads_naturally_at_both_scales():
    assert _humanize(timedelta(hours=17, minutes=3)) == "17h03m"
    assert _humanize(timedelta(minutes=42)) == "42m"
