"""One token-anchored rule for "does this number appear in that text".

docs/code_review.md, Medium #6: the `calculate` provenance check matched by
substring, so a declared input counted as retrieved whenever its digits sat
inside a larger number — the exact weakness citation_verifier's docstring
records fixing for memos. Replayed on 30 real fundamentals provenance files
(1,285 declared calculate inputs), the switch changed exactly one verdict:
a value the model itself labelled "placeholder", which had passed only as a
substring.
"""

from __future__ import annotations

import pytest

import app.agent.tools as tools
from app.application.number_matching import raw_form, value_in_text, value_spans


@pytest.mark.parametrize("value,text", [
    (25.0, "Revenue grew in fiscal 2025."),          # inside a year
    (7.4, "Operating margin was 17.45%."),          # inside a larger decimal
    (420.5, "Total was $3,420.5 thousand."),         # the verifier's own example
    (64896.464, "Revenue was $64,896,464 thousand."),  # hand-converted units
])
def test_digits_inside_a_larger_number_do_not_count(value, text):
    assert not value_in_text(value, text)


@pytest.mark.parametrize("value,text", [
    (11384.0, "Net sales were €11,384.0 million."),   # comma-grouped, trailing .0
    (1364.0, "SBC was $1,364.1 million."),              # integer truncation
    (7.4, "Margin was 7.36%."),                         # rounding at displayed precision
    (10874.36, "FCF: 10,874.360"),                      # trailing zero
    (0.712, "sim=0.712"),
])
def test_legitimate_restatements_still_match(value, text):
    assert value_in_text(value, text)


def test_spans_point_at_the_whole_token():
    text = "FY2025 revenue 69,673 vs 64,896"
    [(start, end)] = value_spans(69673.0, text)
    assert text[start:end] == "69,673"


def test_raw_form_keeps_the_precision_a_model_would_write():
    assert raw_form(11474.0) == "11474"
    assert raw_form(64896.464) == "64896.464"


# ---------------------------------------------------------------------------
# The calculate guard itself
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus():
    tools.reset_run_provenance()
    tools.record_tool_output(
        "FY2025 revenue was $69,673 million, up from $64,896 million in FY2024. "
        "Fiscal 2025 operating margin was 15.6%."
    )
    yield
    tools.reset_run_provenance()


def _input(value, label="x", period="FY2025"):
    return {"value": value, "label": label, "fiscal_period": period,
            "source": "ACN 10-K 2025 §Item 7", "unit": "millions"}


def test_calculate_accepts_a_figure_that_was_retrieved(corpus):
    assert tools.validate_calculate_inputs(
        "(69673 - 64896) / 64896 * 100",
        [_input(69673, "revenue"), _input(64896, "revenue", "FY2024")],
    ) is None


def test_calculate_rejects_a_figure_found_only_inside_a_year(corpus):
    """25 appears in the corpus only as part of "2025". Under the old
    substring rule it passed as retrieved."""
    err = tools.validate_calculate_inputs("25 * 2", [_input(25, "placeholder")])
    assert err and err.startswith("Rejected: no tool returned these figures")


def test_calculate_rejects_a_figure_found_only_inside_a_larger_decimal(corpus):
    err = tools.validate_calculate_inputs("5.6 * 2", [_input(5.6, "margin")])
    assert err and "no tool returned" in err
