"""Tests for the provider abstraction.

The translation layer is the whole risk of this change: everything above it
still speaks Anthropic's message shape, so a mistranslation does not fail
loudly at the boundary — it produces a subtly wrong request, or a cost
number that is wrong in the direction of "cheaper than it was".
"""

import json
from types import SimpleNamespace

import pytest

from app.infrastructure.llm import client as client_mod
from app.infrastructure.llm.client import (
    ProviderNotConfigured,
    RoutingClient,
    get_client,
    resolve_provider,
)
from app.infrastructure.llm.openai_compat import (
    OpenAICompatClient,
    _translate_messages,
    _translate_tool_choice,
    _translate_tools,
    _translate_usage,
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("claude-sonnet-5", "anthropic"),
        ("deepseek-v4-flash", "deepseek"),
        ("deepseek-v4-pro", "deepseek"),
        ("gpt-4.1-nano", "openai"),
        # An id nothing matches must not silently pick a paid third party.
        ("some-unknown-model", "anthropic"),
        (None, "anthropic"),
    ],
)
def test_provider_is_resolved_from_the_model_id(monkeypatch, model, expected):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve_provider(model).name == expected


def test_llm_provider_env_overrides_the_prefix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    # The point of the override: a gateway serving a model under a name whose
    # prefix would otherwise route it somewhere else.
    assert resolve_provider("claude-haiku-4-5-20251001").name == "deepseek"


def test_unknown_llm_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    with pytest.raises(ProviderNotConfigured, match="not a known provider"):
        resolve_provider("deepseek-v4-flash")


def test_missing_key_fails_at_construction_not_at_first_call(monkeypatch):
    """The whole reason the check is eager — a 401 discovered mid-run has
    already been paid for."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ProviderNotConfigured, match="DEEPSEEK_API_KEY"):
        get_client("deepseek-v4-flash")


def test_get_client_with_no_model_routes_per_request(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    routing = get_client()
    assert isinstance(routing, RoutingClient)

    seen = []

    class Fake:
        def __init__(self, name):
            self.messages = SimpleNamespace(create=self._create)
            self._name = name

        async def _create(self, **kwargs):
            seen.append((self._name, kwargs["model"]))
            return "ok"

    monkeypatch.setattr(client_mod, "get_client", lambda m: Fake(m))

    import asyncio

    asyncio.run(routing.messages.create(model="claude-sonnet-5"))
    asyncio.run(routing.messages.create(model="deepseek-v4-flash"))
    asyncio.run(routing.messages.create(model="claude-sonnet-5"))

    assert seen == [
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("deepseek-v4-flash", "deepseek-v4-flash"),
        ("claude-sonnet-5", "claude-sonnet-5"),
    ]
    # Two models, two sub-clients — the third call reused the first.
    assert len(routing._by_model) == 2


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def test_system_blocks_collapse_to_one_system_message():
    out = _translate_messages(
        [
            {"type": "text", "text": "role instructions"},
            {"type": "text", "text": "evidence pack", "cache_control": {"type": "ephemeral"}},
        ],
        [{"role": "user", "content": "go"}],
    )
    assert out[0] == {"role": "system", "content": "role instructions\n\nevidence pack"}
    assert out[1] == {"role": "user", "content": "go"}
    # cache_control has no target in this dialect and must not leak into the
    # request body as an unknown key.
    assert "cache_control" not in json.dumps(out)


def test_tool_use_becomes_tool_calls_and_tool_result_becomes_a_tool_message():
    out = _translate_messages(
        None,
        [
            {"role": "user", "content": "analyze AVGO"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "tu_1", "name": "ask_edgar",
                     "input": {"ticker": "AVGO"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "revenue 51.6B"},
                ],
            },
        ],
    )
    assistant = out[1]
    assert assistant["content"] == "checking"
    assert assistant["tool_calls"][0]["id"] == "tu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "ask_edgar"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"ticker": "AVGO"}
    assert out[2] == {"role": "tool", "tool_call_id": "tu_1", "content": "revenue 51.6B"}


def test_tool_only_assistant_turn_sends_null_content_not_empty_string():
    """Some OpenAI-compatible servers 400 on `content: ""` beside tool_calls."""
    out = _translate_messages(
        None,
        [{"role": "assistant",
          "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]}],
    )
    assert out[0]["content"] is None


def test_tool_results_precede_free_text_in_the_same_user_turn():
    """The API requires the messages answering a tool call to come first."""
    out = _translate_messages(
        None,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "now write the memo"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "data"},
            ],
        }],
    )
    assert [m["role"] for m in out] == ["tool", "user"]


def test_response_blocks_replayed_into_the_next_request_translate_too():
    """The synthesis and debate schema-retries append `response.content`
    verbatim, so the translator sees objects, not dicts."""
    from app.infrastructure.llm.openai_compat import TextBlock, ToolUseBlock

    out = _translate_messages(
        None,
        [{"role": "assistant",
          "content": [TextBlock(text="hi"), ToolUseBlock(id="x", name="f", input={"a": 1})]}],
    )
    assert out[0]["tool_calls"][0]["id"] == "x"
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


def test_error_tool_results_stay_marked_as_errors():
    out = _translate_messages(
        None,
        [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "is_error": True,
             "content": "did not validate"},
        ]}],
    )
    assert out[0]["content"].startswith("ERROR: ")


def test_tool_schema_translation_carries_strict():
    """`strict` is supported in this dialect and must survive. It was added
    to the debate port's SUBMIT_TOOL because 3 of 3 live turns came back
    flattened without it, costing a retry every time."""
    out = _translate_tools([
        {"name": "submit", "description": "Call once.", "strict": True,
         "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
    ])
    assert out == [{
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Call once.",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            "strict": True,
        },
    }]


def test_a_tool_without_strict_does_not_gain_it():
    out = _translate_tools([{"name": "f", "input_schema": {"type": "object"}}])
    assert "strict" not in out[0]["function"]


def test_forced_single_tool_call_splits_into_two_parameters():
    choice, parallel = _translate_tool_choice(
        {"type": "tool", "name": "submit_argument", "disable_parallel_tool_use": True}
    )
    assert choice == {"type": "function", "function": {"name": "submit_argument"}}
    assert parallel is False


def test_tool_choice_any_and_auto():
    assert _translate_tool_choice({"type": "any"})[0] == "required"
    assert _translate_tool_choice({"type": "auto"})[0] == "auto"
    assert _translate_tool_choice(None) == (None, None)


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------

def _completion(*, content=None, tool_calls=None, finish_reason="stop", usage=None):
    return SimpleNamespace(
        id="cmpl_1",
        model="deepseek-v4-flash",
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
        )],
        usage=usage,
    )


def _shim(thinking_param: str | None = "thinking"):
    """Defaults to a DeepSeek-shaped provider — one that HAS a reasoning
    toggle. Pass None for a provider without one."""
    return OpenAICompatClient(
        api_key="k", base_url="https://example.invalid", provider="deepseek",
        thinking_param=thinking_param,
    )


@pytest.mark.parametrize(
    "finish_reason,stop_reason",
    [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")],
)
def test_stop_reason_mapping(finish_reason, stop_reason):
    """`researcher.py` branches on both mapped values — "max_tokens" to
    recover a memo cut mid-sentence, "tool_use" to continue the loop."""
    resp = _shim()._to_anthropic_shape(_completion(finish_reason=finish_reason), "deepseek-v4-flash")
    assert resp.stop_reason == stop_reason


def test_tool_calls_become_anthropic_tool_use_blocks():
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="submit", arguments='{"stance": "bull"}'),
    )
    resp = _shim()._to_anthropic_shape(
        _completion(content="thinking", tool_calls=[call], finish_reason="tool_calls"),
        "deepseek-v4-flash",
    )
    assert [b.type for b in resp.content] == ["text", "tool_use"]
    assert resp.content[1].id == "call_1"
    assert resp.content[1].input == {"stance": "bull"}


def test_malformed_tool_arguments_degrade_to_the_callers_retry():
    """An empty dict reaches the ports' ValidationError retry; a
    JSONDecodeError would unwind the whole run instead."""
    call = SimpleNamespace(id="c", function=SimpleNamespace(name="submit", arguments="{not json"))
    resp = _shim()._to_anthropic_shape(
        _completion(tool_calls=[call], finish_reason="tool_calls"), "deepseek-v4-flash"
    )
    assert resp.content[0].input == {}


# ---------------------------------------------------------------------------
# Usage — the one that costs money if it is wrong
# ---------------------------------------------------------------------------

def test_cached_prompt_tokens_are_not_counted_twice():
    """DeepSeek's `prompt_tokens` is hits PLUS misses. Pricing that as
    `input` while ALSO counting the hits as `cache_read` overstates every
    cached run."""
    usage = _translate_usage(SimpleNamespace(
        prompt_tokens=1000,
        prompt_cache_hit_tokens=800,
        prompt_cache_miss_tokens=200,
        completion_tokens=300,
    ))
    assert usage.input_tokens == 200
    assert usage.cache_read_input_tokens == 800
    assert usage.output_tokens == 300
    assert usage.input_tokens + usage.cache_read_input_tokens == 1000
    # Nothing is billed to populate the cache in this dialect.
    assert usage.cache_creation_input_tokens == 0


def test_usage_falls_back_to_subtraction_when_the_miss_count_is_absent():
    usage = _translate_usage(SimpleNamespace(
        prompt_tokens=1000,
        prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        completion_tokens=50,
    ))
    assert usage.input_tokens == 600
    assert usage.cache_read_input_tokens == 400


def test_usage_with_no_cache_reporting_at_all():
    usage = _translate_usage(SimpleNamespace(prompt_tokens=120, completion_tokens=30))
    assert (usage.input_tokens, usage.cache_read_input_tokens) == (120, 0)


def test_shim_usage_is_readable_by_the_shared_TokenUsage_type():
    """The accounting sites read Anthropic's field names off `.usage`
    directly; the shim has to answer to those names."""
    from app.domain.token_usage import TokenUsage

    shim_usage = _translate_usage(SimpleNamespace(
        prompt_tokens=100, prompt_cache_hit_tokens=40,
        prompt_cache_miss_tokens=60, completion_tokens=20,
    ))
    converted = TokenUsage.from_response(shim_usage)
    assert converted.input_tokens == 60
    assert converted.cache_read_tokens == 40
    assert converted.output_tokens == 20


# ---------------------------------------------------------------------------
# Cost config
# ---------------------------------------------------------------------------

def test_deepseek_is_priced_so_the_budget_guards_can_fire():
    """An unpriced model makes every turn cost None, which sums to 0.00 and
    can never trip the per-run budget assertion."""
    from app.infrastructure.llm.pricing import MODEL_PRICING

    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        assert set(MODEL_PRICING[model]) == {"input", "output", "cache_write", "cache_read"}


def test_retired_deepseek_ids_are_gone_rather_than_priced():
    """deepseek-chat and deepseek-reasoner no longer exist. Leaving them in
    the table would let a stale .env price cleanly all the way to a 404 —
    warn_if_unpriced is the only thing that would have flagged it."""
    from app.infrastructure.llm.pricing import MODEL_PRICING

    assert "deepseek-chat" not in MODEL_PRICING
    assert "deepseek-reasoner" not in MODEL_PRICING


def test_pricing_overrides_are_merged_from_the_environment(monkeypatch):
    import importlib

    from app.infrastructure.llm import pricing

    monkeypatch.setenv(
        "LLM_PRICING_OVERRIDES",
        json.dumps({"my-gateway-model": {
            "input": 1.5, "output": 2.5, "cache_write": 0.0, "cache_read": 0.1,
        }}),
    )
    reloaded = importlib.reload(pricing)
    try:
        assert reloaded.MODEL_PRICING["my-gateway-model"]["output"] == 2.5
    finally:
        monkeypatch.delenv("LLM_PRICING_OVERRIDES", raising=False)
        importlib.reload(pricing)


def test_incomplete_pricing_override_raises_rather_than_silently_keeping_the_old_rate(monkeypatch):
    import importlib

    from app.infrastructure.llm import pricing

    monkeypatch.setenv("LLM_PRICING_OVERRIDES", json.dumps({"x": {"input": 1.0}}))
    try:
        with pytest.raises(ValueError, match="missing"):
            importlib.reload(pricing)
    finally:
        monkeypatch.delenv("LLM_PRICING_OVERRIDES", raising=False)
        importlib.reload(pricing)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_max_tokens_is_clamped_to_the_providers_ceiling(monkeypatch):
    """Both V4 models top out at 384K, so the clamp does not fire in this
    project today — but it did at 8192 on the retired deepseek-chat, where
    an unclamped 16000 was a 400 on the final turn of an already-paid-for
    run. Tested at a low ceiling so the mechanism stays covered."""
    sent = {}

    shim = OpenAICompatClient(
        api_key="k", base_url="https://example.invalid",
        max_output_tokens=8192, provider="deepseek", thinking_param="thinking",
    )

    async def fake_create(**kwargs):
        sent.update(kwargs)
        return _completion(content="ok")

    monkeypatch.setattr(shim._client.chat.completions, "create", fake_create)
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=16000,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert sent["max_tokens"] == 8192


@pytest.fixture
def capture(monkeypatch):
    """Capture the request the shim would send, without sending it."""
    sent = {}
    shim = _shim()

    async def fake_create(**kwargs):
        sent.clear()
        sent.update(kwargs)
        return _completion(content="ok")

    monkeypatch.setattr(shim._client.chat.completions, "create", fake_create)
    return shim, sent


@pytest.mark.anyio
async def test_anthropic_only_parameters_are_dropped(capture):
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        betas=["some-beta"],
        top_k=5,
    )
    assert "betas" not in sent and "top_k" not in sent


@pytest.mark.anyio
async def test_forcing_a_tool_disables_thinking(capture):
    """Live 400: "Thinking mode does not support this tool_choice". Thinking
    is ON by default here, so a forced tool call fails unless it is turned
    off explicitly — and the ports force a tool on every structured call."""
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "f", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "f", "disable_parallel_tool_use": True},
    )
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert sent["parallel_tool_calls"] is False


@pytest.mark.anyio
async def test_tool_choice_required_also_disables_thinking(capture):
    """`required` is rejected by thinking mode too — only `auto` survives."""
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "f", "input_schema": {"type": "object"}}],
        tool_choice={"type": "any"},
    )
    assert sent["tool_choice"] == "required"
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.anyio
async def test_auto_tool_choice_keeps_thinking(capture):
    """`auto` does not constrain the model, so thinking survives and the
    caller's requested effort carries through."""
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "f", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
    )
    assert sent["extra_body"] == {"thinking": {"type": "enabled", "reasoning_effort": "low"}}


@pytest.mark.anyio
async def test_no_thinking_parameter_means_thinking_off(capture):
    """The regression this exists to prevent.

    The internal shape is Anthropic's, where sending no `thinking` gets
    none. DeepSeek defaults the other way, so leaving its default alone
    turned reasoning ON for every call that does not force a tool — and
    reasoning bills as output against `max_tokens`. Live: all 2 ACN digest
    batches came back unparseable and ended the run, 15 of 16 on MSFT, and
    the technical node (max_tokens=512) returned an empty interpretation.
    """
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.anyio
async def test_a_provider_with_no_reasoning_toggle_gets_no_extra_body(monkeypatch):
    """Writing an unknown key into extra_body is a 400 on a strict server,
    so the toggle is only sent to providers that document one."""
    sent = {}
    shim = _shim(thinking_param=None)

    async def fake_create(**kwargs):
        sent.update(kwargs)
        return _completion(content="ok")

    monkeypatch.setattr(shim._client.chat.completions, "create", fake_create)
    await shim.messages.create(
        model="gpt-4.1-nano", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
    )
    assert "extra_body" not in sent


@pytest.mark.anyio
async def test_an_effort_the_provider_does_not_have_is_dropped_not_rounded(capture):
    """Anthropic's "medium" has no counterpart; rounding it to a neighbour
    would silently change what the caller asked for."""
    shim, sent = capture
    await shim.messages.create(
        model="deepseek-v4-flash", max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.fixture
def anyio_backend():
    return "asyncio"
