"""
`ask_edgar` can restrict retrieval to named Items.

Why it exists: a filing says similar things in two places. Management's own
ICFR conclusion is in Item 9A; the auditor's near-identical opinion on the
same subject is in the financial statements. Measured 2026-09-12 on ACN's
re-chunked FY2025 10-K, ranking the Item 9A chunk that says "our management
concluded that our internal control over financial reporting was effective":

  "Did management conclude that ICFR was effective ...?"          rank 1
  the agent's "Accenture plc FY2025: In Item 9A, what exact
    conclusion does management state about ... effectiveness
    and material weaknesses?"                                     not in top 8
  that same question + section_path_contains=["Item 9A"]          rank 1

So naming the Item in the question text does not narrow retrieval — it only
dilutes the question. The filter does.
"""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

import app.agent.tools as tools
import app.main as main


def _schema() -> dict:
    return next(t for t in tools.TOOLS if t["name"] == "ask_edgar")["input_schema"]


# ---------------------------------------------------------------------------
# The tool surface
# ---------------------------------------------------------------------------

def test_the_tool_takes_a_sections_argument():
    props = _schema()["properties"]
    # `_strictify` lists every property as required-but-nullable, so an
    # optional one arrives as an explicit null rather than an absent key —
    # which is why the dispatch reads it as `inputs.get("sections") or None`
    # and not `inputs.get("sections", None)`. See the `limit: null` bug.
    assert props["sections"]["type"] == ["array", "null"]
    assert props["sections"]["items"]["type"] == "string"


def test_the_description_says_naming_the_item_in_the_question_does_not_work():
    description = next(t for t in tools.TOOLS if t["name"] == "ask_edgar")["description"]
    assert "sections" in description
    assert "does NOT" in description


# ---------------------------------------------------------------------------
# What reaches the server
# ---------------------------------------------------------------------------

class _CapturingResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, sent: dict):
        self._sent = sent

    def json(self) -> dict:
        return {"answer": "a", "citations": [], "chunks": []}


@pytest.fixture
def sent(monkeypatch) -> dict:
    """Capture the JSON body ask_edgar POSTs to /ask."""
    captured: dict = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            captured.update(json or {})
            return _CapturingResponse(captured)

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda *a, **k: _Client())
    monkeypatch.setattr(tools, "_record_delegated_usage", lambda resp: None)
    tools.reset_run_provenance()
    return captured


@pytest.mark.anyio
async def test_sections_are_sent_as_the_section_filter(sent):
    await tools._dispatch("ask_edgar", {"question": "q", "sections": ["Item 9A"]})
    assert sent["section_path_contains"] == ["Item 9A"]


@pytest.mark.anyio
async def test_no_sections_means_no_filter(sent):
    await tools._dispatch("ask_edgar", {"question": "q"})
    assert sent["section_path_contains"] is None


@pytest.mark.anyio
async def test_an_empty_section_list_is_not_a_filter_that_matches_nothing(sent):
    """A model that means "search everything" sends [] about as often as it
    omits the key. Passed through, that would filter out the whole corpus."""
    await tools._dispatch("ask_edgar", {"question": "q", "sections": []})
    assert sent["section_path_contains"] is None


# ---------------------------------------------------------------------------
# A filter naming something the corpus doesn't have
# ---------------------------------------------------------------------------

class _StubChunkRepo:
    def __init__(self, known: set[str]):
        self._known = known
        self.asked_for: list[str] | None = None

    async def sections_with_content(self, sections, tickers=None):
        self.asked_for = sections
        return {s for s in sections if s in self._known}


def _stub_ask(monkeypatch, repo, seen_filters: list):
    class _Retrieval:
        def __init__(self, **kw):
            pass

        async def retrieve_full(self, question, k=8, filters=None):
            seen_filters.append(filters.section_path_contains)
            return [], _Decomposition()

    class _Decomposition:
        usage = None
        was_decomposed = False

    class _Answer:
        answer = "no excerpts"
        citations: list = []
        usage = None

    class _Report:
        unverified: list = []

        def summary(self):
            return ""

    monkeypatch.setattr(main, "ChunkRepository", lambda: repo)
    monkeypatch.setattr(main, "RetrievalService", _Retrieval)
    monkeypatch.setattr(main, "_embedder", lambda: None)
    monkeypatch.setattr(main, "_decomposer", lambda: type("D", (), {"model": "m"})())
    monkeypatch.setattr(main, "_report_usage", lambda *a, **k: None)
    monkeypatch.setattr(main, "verify_answer", lambda *a, **k: _Report())

    async def _answer(**kw):
        return _Answer()

    monkeypatch.setattr(main, "answer_question", _answer)


def test_an_unmatched_section_is_reported_and_the_filter_dropped(monkeypatch):
    """Retrieving nothing produces an empty answer, which reads to the caller
    as "the filing doesn't say" rather than "you named a section that isn't
    there" — and costs one of 30 budgeted calls to learn nothing."""
    repo = _StubChunkRepo(known=set())
    seen: list = []
    _stub_ask(monkeypatch, repo, seen)

    with TestClient(main.app) as client:
        r = client.post("/ask", json={"question": "q", "section_path_contains": ["Item 99Z"]})

    assert r.status_code == 200
    assert r.json()["dropped_section_filter"] == ["Item 99Z"]
    assert seen == [None], "the retrieval should have run without the filter"


def test_a_matched_section_is_kept(monkeypatch):
    repo = _StubChunkRepo(known={"Item 9A"})
    seen: list = []
    _stub_ask(monkeypatch, repo, seen)

    with TestClient(main.app) as client:
        r = client.post("/ask", json={"question": "q", "section_path_contains": ["Item 9A"]})

    assert r.json()["dropped_section_filter"] is None
    assert seen == [["Item 9A"]]


@pytest.mark.anyio
async def test_the_agent_is_told_when_its_filter_was_dropped(monkeypatch):
    """Otherwise it reads a whole-filing answer as one from the section it
    asked for, and attributes the figures to the wrong Item."""

    class _Resp:
        status_code = 200
        headers: dict = {}

        def json(self):
            return {
                "answer": "a",
                "citations": [],
                "chunks": [],
                "dropped_section_filter": ["Item 99Z"],
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            return _Resp()

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda *a, **k: _Client())
    monkeypatch.setattr(tools, "_record_delegated_usage", lambda resp: None)
    tools.reset_run_provenance()

    out = await tools._dispatch("ask_edgar", {"question": "q", "sections": ["Item 99Z"]})
    assert "Item 99Z" in out
    assert "whole filing" in out


def test_the_tool_descriptions_point_at_the_list_of_valid_section_names():
    """Given the filter but no list of names, the agent guesses note titles:
    18 of 29 filtered calls on ACN named something no chunk is filed under."""
    by_name = {t["name"]: t for t in tools.TOOLS}

    assert "sections_available" in by_name["check_corpus"]["description"]

    ask = by_name["ask_edgar"]
    assert "sections_available" in ask["description"]
    sections = ask["input_schema"]["properties"]["sections"]["description"]
    assert "sections_available" in sections
    assert "Revenue Recognition" in sections, "name a concrete wrong guess"
