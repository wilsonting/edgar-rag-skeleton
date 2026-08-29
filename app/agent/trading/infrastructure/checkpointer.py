import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Best-effort only, and deliberately placed above the langgraph imports:
# langgraph.checkpoint.serde._msgpack freezes STRICT_MSGPACK_ENABLED into a
# module-level constant at import time, so setting this afterwards does
# nothing. Even here it wins the race only if this module is imported before
# anything else pulls in langgraph — graph.py's `from langgraph.graph import
# ...` triggers the same constant. To make strict mode hold process-wide,
# export LANGGRAPH_STRICT_MSGPACK=true in the environment before Python starts.
#
# This flag is NOT what protects this checkpointer. It only changes the default
# for serializers built without an explicit allowlist; build_serde() passes
# allowed_msgpack_modules directly, which enforces the list either way.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from contextlib import asynccontextmanager  # noqa: E402

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from psycopg_pool import AsyncConnectionPool  # noqa: E402

DB_URI = os.getenv("TRADING_CHECKPOINT_DB_URI")

connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

# Every custom domain type that lands in TradingState needs an entry here —
# add to this list as Phase 2+ introduces FundamentalsReport, TechnicalReport, etc.
ALLOWED_MSGPACK_MODULES = [
    ("app.agent.trading.domain.decision_memo", "Verdict"),
    ("app.agent.trading.domain.decision_memo", "DecisionMemo"),
    ("app.agent.trading.domain.fundamentals_report", "FundamentalsReport"),
    ("app.agent.trading.domain.technical_report", "TechnicalReport"),
    ("app.agent.trading.domain.technical_report", "TechnicalIndicators"),
    ("app.agent.trading.domain.news_digest", "NewsItem"),
    ("app.agent.trading.domain.news_digest", "NewsDigest"),
    ("app.agent.trading.domain.news_digest", "SentimentSummary"),
    # Phase 5. Three entries for one channel because DebateTurn nests
    # DebateTurnPayload nests DebateClaim, and an unregistered type fails on
    # DESERIALIZATION ONLY — a live run stays green and the resume goes red,
    # which is the worst possible place to discover a missing line.
    ("app.agent.trading.domain.debate", "DebateClaim"),
    ("app.agent.trading.domain.debate", "DebateTurnPayload"),
    ("app.agent.trading.domain.debate", "DebateTurn"),
    # Phase 6. Three entries, same reason as the debate ones above: RiskTurn
    # nests RiskTurnPayload nests RiskFactor/RiskScore, and a missing entry
    # fails on DESERIALIZATION ONLY. RiskLedgerEntry is deliberately absent —
    # it never enters state (domain/risk.py's docstring on why) — so if it is
    # ever persisted, that line belongs in the same commit as the change that
    # persists it, not here.
    ("app.agent.trading.domain.risk", "RiskFactor"),
    ("app.agent.trading.domain.risk", "RiskScore"),
    ("app.agent.trading.domain.risk", "RiskTurnPayload"),
    ("app.agent.trading.domain.risk", "RiskTurn"),
    # Phase 8. RunTermination is a plain str Enum (same registration need as
    # Verdict above) — RunBudget and CostEvent are BaseModels like every
    # other entry here.
    ("app.agent.trading.domain.budget", "RunBudget"),
    ("app.agent.trading.domain.budget", "CostEvent"),
    ("app.agent.trading.domain.budget", "RunTermination"),
]


def build_serde() -> JsonPlusSerializer:
    """The serializer this checkpointer actually uses.

    Exposed so tests exercise the real configuration instead of constructing
    their own equivalent — a test that builds its own JsonPlusSerializer would
    keep passing even if this module stopped passing the allowlist, which is
    precisely the regression worth catching.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def build_checkpointer():
    async with AsyncConnectionPool(
        conninfo=DB_URI, max_size=10, kwargs=connection_kwargs, open=False
    ) as pool:
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool, serde=build_serde())
        await checkpointer.setup()  # idempotent — creates tables once
        yield checkpointer