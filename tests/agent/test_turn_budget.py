"""Phase 9 measured 2 of 3 fundamentals runs hitting MAX_TURNS exactly and
ending on "[MAX_TURNS reached — forcing memo from gathered data]". The agent
had no idea the budget existed, so it was writing its memo under a
guillotine rather than to a deadline.

The cap itself is unchanged. What changed is that the agent is told the
number up front and counted down near the end — the same treatment
ask_edgar's budget got, for the same reason: an unannounced limit truncates
wherever the agent happens to be standing.
"""

from __future__ import annotations

import re
from pathlib import Path

import app.agent.researcher as researcher

SOURCE = Path(researcher.__file__).read_text()


def test_the_task_message_states_the_turn_budget():
    """Stated up front, not only warned about near the end, so the agent can
    plan the 12-item checklist against it."""
    assert 'You have {MAX_TURNS} tool-calling turns' in SOURCE


def test_the_budget_is_in_the_task_not_the_system_prompt():
    """The system block carries its own cache breakpoint and is identical
    across every run. Interpolating a runtime number into it would rewrite
    that cache whenever the config moved."""
    system_block = SOURCE[SOURCE.index('"text": system_prompt'):]
    assert "MAX_TURNS" not in system_block[:200]


def test_the_warn_band_is_narrow():
    """Every warning lands in the conversation the agent re-reads on all
    subsequent turns, and tool results are the provenance corpus the
    containment guards scan. A countdown on all 45 turns would be 45 copies
    of a changing number in that corpus."""
    assert researcher.TURN_WARN_AT <= 10
    assert researcher.TURN_WARN_AT < researcher.MAX_TURNS


def test_the_countdown_fires_only_inside_the_band():
    guard = re.search(r"remaining = MAX_TURNS - \(turn \+ 1\)\s*\n\s*if (.+):", SOURCE)
    assert guard, "the warn-band guard moved; this test needs updating"
    condition = guard.group(1)
    # Must exclude zero: at zero remaining the loop exits into the existing
    # forced-memo path, which already says its piece.
    assert "0 < remaining" in condition
    assert "TURN_WARN_AT" in condition


def test_tool_results_still_come_first_in_the_user_message():
    """The API requires tool_result blocks at the head of the user message;
    a trailing text block is legal after them. Appending the notice the
    other way round would 400 the whole run."""
    block = SOURCE[SOURCE.index("content = list(tool_results)"):]
    append_index = block.index("content.append(")
    assert append_index > 0
    assert "messages.append" in block[append_index:]


def test_the_forced_memo_path_still_exists():
    """The warn band is a nudge, not a replacement — a run that ignores it
    must still terminate with a memo rather than an exception."""
    assert "MAX_TURNS reached" in SOURCE
    assert "Write the memo now" in SOURCE
