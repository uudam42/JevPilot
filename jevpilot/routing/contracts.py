"""Adapter contracts for model-backed routers.

Routing semantics (requests, decisions, validation) live in :mod:`jevpilot.routing`.
Provider mechanics live in adapters that implement these protocols. Adapters
depend on these contracts, and routing never depends on a concrete adapter:

    LLMRouter ──► LLMAdapter            text in / text out (prompt built by the router)
    JevRouter ──► RoutingModelAdapter   RoutingRequest in / raw decision out

Any provider-specific formatting belongs inside the adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from jevpilot.core import FrozenModel, ModelUsage
from jevpilot.routing.request import RoutingRequest


class RoutingModelResponse(FrozenModel):
    """Raw, untrusted model output for one routing request.

    ``output`` is either text (expected to contain a JSON decision) or an
    already-decoded JSON object. Either way it is parsed and validated before
    it can become a :class:`~jevpilot.core.RoutingDecision`.
    """

    output: Any
    model: str | None = None
    usage: ModelUsage | None = None  # None: the provider reported nothing
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RoutingModelAdapter(Protocol):
    """A model that consumes a :class:`RoutingRequest` directly (e.g. Jev)."""

    def infer(
        self, request: RoutingRequest, *, timeout_s: float | None = None
    ) -> RoutingModelResponse: ...

    def describe(self) -> dict[str, Any]:
        """Provider, model identifier and sampling configuration, for run records."""
        ...


class LLMPrompt(FrozenModel):
    """Provider-neutral prompt built by :mod:`jevpilot.routing.prompts`."""

    system: str
    user: str
    response_schema: dict[str, Any]
    version: str


class LLMCompletion(FrozenModel):
    text: str
    model: str | None = None
    usage: ModelUsage | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMAdapter(Protocol):
    """A text-completion model. Knows nothing about routing."""

    def complete(self, prompt: LLMPrompt, *, timeout_s: float | None = None) -> LLMCompletion: ...

    def describe(self) -> dict[str, Any]: ...
