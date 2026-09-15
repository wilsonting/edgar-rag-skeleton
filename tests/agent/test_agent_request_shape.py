"""Every agent turn sends the same request, and three failures that read as
success.

**The request shape.** The cached prefix is tools + system + messages. The
loop sent `tools` and a cacheable system block on every turn — except the
two that write the memo: the `max_tokens` continuation, and the forced-memo
call at the end, which sent no tools and a bare string system prompt.

That last one carries the WHOLE conversation, and `tools.py` records that
Phase 9 measured 2 of 3 fundamentals runs ending on it. So the largest
request of the run was the one guaranteed to miss the cache.

It matters more than it looks on the providers this project actually runs.
On DeepSeek and luna, caching is automatic PREFIX matching and
`cache_control` is stripped in translation (`openai_compat._flatten_text`),
so a byte-identical prefix is the only thing that can produce a hit at all —
and dropping the tools block changes the request at position zero.

`tool_choice: none` keeps the tools in the prefix while forbidding a call,
which is what those turns need: the prompt already says "do not call any
more tools", and this makes it true rather than hoped for.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

import app.agent.researcher as researcher
from app.domain.token_usage import response_text


def _usage():
    return SimpleNamespace(
        input_tokens=1, cache_creation_input_tokens=0,
        cache_read_input_tokens=0, output_tokens=1,
    )


class _Client:
    """Answers with a memo the moment it is asked for one."""

    def __init__(self, turns_before_memo: int = 0):
        self.requests: list[dict] = []
        self._turns = turns_before_memo

    @property
    def messages(self):
        return self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="# memo\n## Assessment\nok")],
            stop_reason="end_turn", usage=_usage(),
        )


@pytest.fixture
def client(monkeypatch):
    c = _Client()
    monkeypatch.setattr(researcher, "get_client", lambda model: c)
    return c


# ---------------------------------------------------------------------------
# The request shape
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_every_call_sends_the_tools_block(client, monkeypatch):
    """Dropping it changes the request at position zero, so no prefix can
    match — on the compat providers that is the whole cache."""
    monkeypatch.setattr(researcher, "MAX_TURNS", 1)
    await researcher.run_agent("task", "system")
    assert client.requests, "no model call was made"
    for req in client.requests:
        assert req["tools"] is researcher.TOOLS


@pytest.mark.anyio
async def test_every_call_sends_a_cacheable_system_block(client, monkeypatch):
    """Not a bare string — the forced-memo call used to send one."""
    monkeypatch.setattr(researcher, "MAX_TURNS", 1)
    await researcher.run_agent("task", "system")
    for req in client.requests:
        system = req["system"]
        assert isinstance(system, list), system
        assert system[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.anyio
async def test_every_call_carries_exactly_one_message_breakpoint(client, monkeypatch):
    monkeypatch.setattr(researcher, "MAX_TURNS", 1)
    await researcher.run_agent("task", "system")
    for req in client.requests:
        marked = [
            b for m in req["messages"]
            if isinstance(m["content"], list)
            for b in m["content"]
            if isinstance(b, dict) and "cache_control" in b
        ]
        assert len(marked) == 1, marked


@pytest.mark.anyio
async def test_the_memo_turns_are_forbidden_from_calling_a_tool(monkeypatch):
    """The tools stay in the prefix; tool_choice stops them being used."""
    calls: list[dict] = []

    class _ToolThenMemo(_Client):
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                block = SimpleNamespace(
                    type="tool_use", id="t1", name="check_corpus", input={"ticker": "ACN"},
                )
                return SimpleNamespace(
                    content=[block], stop_reason="tool_use", usage=_usage()
                )
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="# memo\n## Assessment\nok")],
                stop_reason="end_turn", usage=_usage(),
            )

    c = _ToolThenMemo()
    monkeypatch.setattr(researcher, "get_client", lambda model: c)
    monkeypatch.setattr(researcher, "MAX_TURNS", 1)

    async def fake_execute(name, inputs):
        return "ok"

    monkeypatch.setattr(researcher, "execute_tool", fake_execute)

    await researcher.run_agent("task", "system")

    # Turn 1 may call tools; the forced memo may not.
    assert "tool_choice" not in calls[0]
    assert calls[-1]["tool_choice"] == {"type": "none"}


# ---------------------------------------------------------------------------
# `--test` no longer crashes after the run
# ---------------------------------------------------------------------------

def test_the_test_flag_assigns_a_mode(monkeypatch):
    """`mode` was unbound on this branch, so `mode != "test"` raised
    UnboundLocalError after the run had already completed and been paid for."""
    import inspect
    source = inspect.getsource(researcher.main)
    test_branch = source.split("if args.test:")[1].split("elif")[0]
    assert 'mode = "test"' in test_branch


# ---------------------------------------------------------------------------
# A response with no content blocks
# ---------------------------------------------------------------------------

def test_response_text_of_an_empty_completion_is_empty_not_an_indexerror():
    """The compat adapter returns no blocks when the provider sends no text,
    which is what a completion cut off at the token limit looks like."""
    assert response_text(SimpleNamespace(content=[])) == ""
    assert response_text(SimpleNamespace(content=None)) == ""
    assert response_text(SimpleNamespace()) == ""


def test_response_text_joins_every_text_block_not_just_the_first():
    blocks = [
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="tool_use", id="1", name="x", input={}),
        SimpleNamespace(type="text", text="b"),
    ]
    assert response_text(SimpleNamespace(content=blocks)) == "ab"


@pytest.mark.anyio
async def test_ask_returns_an_explicit_empty_answer_rather_than_a_500(monkeypatch):
    from app.domain.chunk import Chunk
    from app.infrastructure.repositories.chunk_repo import RetrievedChunk
    import app.llm as llm

    chunk = Chunk(
        id=1, section_id=1, content="Revenue was $64.9 billion.", chunk_index=0,
        token_count=5, ticker="ACN", filed_date=date(2025, 10, 9),
        filing_type="10-K", section_path=["Part II", "Item 8"],
    )

    class _Empty:
        @property
        def messages(self):
            return self

        async def create(self, **kwargs):
            return SimpleNamespace(
                content=[], stop_reason="max_tokens",
                usage=SimpleNamespace(
                    input_tokens=1, output_tokens=0,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0,
                ),
            )

    monkeypatch.setattr(llm, "get_client", lambda model: _Empty())

    result = await llm.answer_question(
        "q", [RetrievedChunk(chunk=chunk, similarity=0.5)], model="m"
    )
    assert "returned no answer" in result.answer
    assert result.citations == []
