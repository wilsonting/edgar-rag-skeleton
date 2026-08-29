"""The Phase 9 battery's defect signature was notation: three memos, three
unit/label errors, no fabrications and no arithmetic errors.

The research agent's own prompt already forbids exactly this and the rule
works — NFLX's fundamentals report says "a 0.41-turn improvement". It was
the SYNTHESIS layer, which had no equivalent rule, that turned that into
"41 basis points". These pin the rule into both synthesis prompts and pin
that they cannot drift apart.
"""

from __future__ import annotations

import pytest

from app.agent.trading.infrastructure import synthesis_port as port

PROMPTS = pytest.mark.parametrize(
    "prompt",
    [port.RESEARCH_MANAGER_SYSTEM, port.RISK_JUDGE_SYSTEM],
    ids=["research_manager", "risk_judge"],
)


@PROMPTS
def test_both_synthesis_prompts_carry_the_units_rule(prompt):
    assert "UNITS SURVIVE RESTATEMENT" in prompt


@PROMPTS
def test_the_rule_names_the_defect_that_was_actually_measured(prompt):
    """A rule stated abstractly ("use correct units") is advice. The measured
    instance is what makes it actionable, and it is the memo's own numbers."""
    assert "0.41-turn" in prompt
    assert "41 basis points" in prompt


@PROMPTS
def test_the_rule_covers_all_three_measured_notation_defects(prompt):
    # NFLX: turns reported as basis points.
    assert "TURNS" in prompt
    # AVGO: a currency sign on a leverage multiple.
    assert "$2.80x" in prompt
    # ACN: a gross ratio presented beside the word "net", unlabelled.
    assert "Gross and net are different figures" in prompt


def test_the_two_prompts_share_one_source_for_the_rule():
    """Two copies would drift. Both interpolate `_UNITS_RULE`, so a change
    reaches the Research Manager and the Risk Judge together."""
    assert port._UNITS_RULE in port.RESEARCH_MANAGER_SYSTEM
    assert port._UNITS_RULE in port.RISK_JUDGE_SYSTEM


@PROMPTS
def test_the_rule_does_not_contradict_rule_one(prompt):
    """Rule 1 forbids restating figures at all, and both models violate it
    routinely — NFLX's memo opens with five numbers in a field where none
    are allowed. This is written as a FALLBACK, not a licence: it has to say
    so, or it reads as permission to restate."""
    rule = prompt[prompt.index("UNITS SURVIVE RESTATEMENT"):]
    assert "Rule 1 says not to restate figures at all" in rule
