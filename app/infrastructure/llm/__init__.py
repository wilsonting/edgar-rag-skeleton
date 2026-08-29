"""Provider-agnostic LLM access.

`get_client(model)` returns a client that duck-types
`anthropic.AsyncAnthropic` regardless of who actually serves the model, so
the rest of the app keeps speaking one message shape. `MODEL_PRICING` is
the cost config every budget guard reads.
"""

from app.infrastructure.llm.client import (
    LLMBadRequestError,
    LLMClient,
    ProviderNotConfigured,
    ProviderSpec,
    RoutingClient,
    get_client,
    is_anthropic,
    resolve_provider,
)
from app.infrastructure.llm.pricing import MODEL_PRICING

__all__ = [
    "LLMBadRequestError",
    "LLMClient",
    "MODEL_PRICING",
    "ProviderNotConfigured",
    "ProviderSpec",
    "RoutingClient",
    "get_client",
    "is_anthropic",
    "resolve_provider",
]
