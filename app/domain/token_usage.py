"""One token-usage shape, shared by everything that spends money.

Exists because the research agent delegates work over HTTP to the FastAPI
app, and until 2026-08-27 that delegated spend was invisible: `/ask`,
`/extract` and the query decomposer each made real Claude calls and none of
them was logged, so `docs/cost-log.jsonl`, `RunBudget` and `--max-usd` saw
only the agent's own loop. Measured on the Phase 9 battery, that was ~28% of
real spend, and it let AVGO exceed its $1.10 cap by 28% without tripping the
guard (see trading-agent-known-gaps.md, "Fundamentals cost audit").

Accounting belongs to the caller that owns the run, not to a stateless HTTP
server: only the caller knows the `run_id`, and only its state feeds
`check_run_guards`. So the server reports what it spent and the client adds
it to the run's ledger — this type is what crosses that boundary.
"""

from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @classmethod
    def from_response(cls, usage) -> "TokenUsage":
        """Read an Anthropic response's `.usage`. The cache fields are absent
        on responses from calls that set no `cache_control`, which is every
        call this covers today — kept anyway so a later caching change is
        picked up without touching this."""
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.input_tokens or self.output_tokens
            or self.cache_write_tokens or self.cache_read_tokens
        )


# Response header the API uses to report what a request spent. A HEADER
# rather than a body field on purpose: the agent reads tool-result bodies
# verbatim into its provenance corpus, and every numeric guard in the
# trading pipeline does containment against that corpus. Adding token counts
# to a body would put four new numbers into evidence for every retrieval —
# numbers that could then "back" a figure in a memo. The header keeps the
# agent-visible bytes identical to what they were.
USAGE_HEADER = "X-LLM-Usage"
