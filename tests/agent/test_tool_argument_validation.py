"""Tool-call arguments: recovered by the validator, prevented by strict.

Both layers exist because of one live failure. A 376-second MSFT run died
at turn ~30 when the model called `check_latest_filings(tickers=["MSFT"])`
— plural, borrowed from `ask_edgar`, which is the only tool that takes a
list. `_dispatch` read `inputs["ticker"]` as a bare subscript, so the
KeyError propagated out of `execute_tool` and ended the run at the tool
call, with the whole run's spend unlogged because cost events are written
on node completion.
"""

import pytest

from app.agent import tools as tool_mod
from app.agent.tools import TOOLS, _strictify, _validate_tool_inputs, execute_tool


# ---------------------------------------------------------------------------
# The validator — recovery
# ---------------------------------------------------------------------------

def test_the_call_that_killed_the_run_is_now_a_message():
    problem = _validate_tool_inputs(
        "check_latest_filings", {"tickers": ["MSFT"], "form_types": ["8-K"]}
    )
    assert problem is not None
    # Both halves matter: naming only the missing key would leave the model
    # guessing what was wrong with the list it did send.
    assert "ticker" in problem
    assert "tickers" in problem


def test_the_message_names_the_arguments_the_tool_accepts():
    """What makes the retry succeed rather than repeat."""
    problem = _validate_tool_inputs("check_latest_filings", {"tickers": ["MSFT"]})
    assert "ticker (string)" in problem
    assert "form_types" in problem and "optional" in problem


def test_a_correct_call_passes():
    assert _validate_tool_inputs("check_latest_filings", {"ticker": "MSFT"}) is None
    assert _validate_tool_inputs("ask_edgar", {"question": "q"}) is None
    assert _validate_tool_inputs("ask_edgar", {"question": "q", "tickers": ["MSFT"]}) is None


def test_an_omitted_optional_argument_is_not_an_error():
    """Strict mode lists every property in `required` on the wire, but
    `form_types` is still genuinely optional and our own validator must not
    start rejecting calls that leave it out."""
    assert _validate_tool_inputs("check_latest_filings", {"ticker": "MSFT"}) is None
    assert _validate_tool_inputs(
        "check_latest_filings", {"ticker": "MSFT", "form_types": None}
    ) is None


def test_a_genuinely_missing_required_argument_is_still_caught():
    problem = _validate_tool_inputs("extract_metrics", {"ticker": "MSFT"})
    assert "fiscal_period" in problem and "filing_type" in problem


def test_an_unknown_tool_is_left_to_dispatch():
    assert _validate_tool_inputs("no_such_tool", {"anything": 1}) is None


@pytest.mark.anyio
async def test_a_bad_call_returns_text_and_never_reaches_dispatch(monkeypatch):
    """The failure has to come back as a tool result the model can read, the
    way every other failure in _dispatch already does."""
    called = False

    async def explode(name, inputs):
        nonlocal called
        called = True
        raise AssertionError("dispatch must not run for an invalid call")

    monkeypatch.setattr(tool_mod, "_dispatch", explode)
    result = await execute_tool("check_latest_filings", {"tickers": ["MSFT"]})
    assert not called
    assert result.startswith("Error: check_latest_filings")


@pytest.mark.anyio
async def test_a_rejected_call_records_nothing_into_the_provenance_corpus(monkeypatch):
    """A rejected call retrieved nothing. The corpus is what every numeric
    guard checks memo figures against, so an error string must not enter it."""
    recorded = []
    monkeypatch.setattr(tool_mod, "record_tool_output", lambda text: recorded.append(text))
    await execute_tool("check_latest_filings", {"tickers": ["MSFT"]})
    assert recorded == []


# ---------------------------------------------------------------------------
# strict — prevention
# ---------------------------------------------------------------------------

def test_every_tool_is_strict_and_strict_compatible():
    """`strict: true` on a schema that is not strict-compatible is a 400 on
    the first call, so the two must never drift apart."""
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool.get("strict") is True, tool["name"]
        assert schema["additionalProperties"] is False, tool["name"]
        assert set(schema["required"]) == set(schema["properties"]), tool["name"]


def test_optional_properties_become_nullable_rather_than_required_for_real():
    """The dispatch code reads optionals through .get(), so an explicit null
    behaves exactly as the previously-absent key did."""
    schema = next(t for t in TOOLS if t["name"] == "check_latest_filings")["input_schema"]
    assert schema["properties"]["form_types"]["type"] == ["array", "null"]
    assert schema["properties"]["ticker"]["type"] == "string"


def test_strictify_closes_nested_objects_too():
    """`calculate.inputs` is an array of objects; strict is recursive."""
    schema = next(t for t in TOOLS if t["name"] == "calculate")["input_schema"]
    item = schema["properties"]["inputs"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_strictify_leaves_a_non_object_schema_alone():
    assert _strictify({"type": "string"}) == {"type": "string"}


def test_strictify_preserves_descriptions_and_enums():
    """Losing these would quietly degrade the model's instructions."""
    schema = next(t for t in TOOLS if t["name"] == "calculate")["input_schema"]
    unit = schema["properties"]["inputs"]["items"]["properties"]["unit"]
    assert "millions" in unit["enum"]
    assert unit["description"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
