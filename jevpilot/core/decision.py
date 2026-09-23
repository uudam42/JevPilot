"""Structured routing decisions and routing diagnostics.

A router answers ``R(S_t, C_t) → D_t``. :class:`RoutingDecision` is ``D_t``.
:class:`RoutingOutcome` wraps it with provider-neutral diagnostics (attempts,
latency, model usage) so decisions from any router can be traced and compared.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from jevpilot.core._base import FrozenModel, new_id
from jevpilot.core.observation import ErrorInfo


class RoutingIntent(StrEnum):
    """What a routing decision asks the loop to do.

    ``INVOKE`` runs a capability. The other intents propose no action and differ
    only in *why*: the router believes the goal is met (``FINISH``), it needs
    information only a human can give (``ASK_HUMAN``), or it has nothing to
    offer (``IDLE``). Asking a human is a legitimate outcome, not an error.
    """

    INVOKE = "invoke"
    FINISH = "finish"
    ASK_HUMAN = "ask_human"
    IDLE = "idle"


class RoutingDecision(FrozenModel):
    """A router's choice of the next capability to invoke.

    ``capability_id=None`` means the router proposes *no* action; ``intent``
    says why. The controller then defers to evaluation/policy.
    """

    id: str = Field(default_factory=lambda: new_id("dec"))
    capability_id: str | None
    inputs: dict[str, Any] = Field(default_factory=dict)
    intent: RoutingIntent = RoutingIntent.INVOKE
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: tuple[str, ...] = ()
    router_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_intent(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("intent") is None:
            data = dict(data)
            data["intent"] = (
                RoutingIntent.IDLE if data.get("capability_id") is None else RoutingIntent.INVOKE
            )
        return data

    @model_validator(mode="after")
    def _intent_matches_capability(self) -> RoutingDecision:
        invokes = self.intent is RoutingIntent.INVOKE
        if invokes != (self.capability_id is not None):
            raise ValueError("intent 'invoke' requires a capability_id; other intents forbid one")
        return self

    @property
    def is_no_action(self) -> bool:
        return self.capability_id is None

    @classmethod
    def no_action(
        cls,
        reason: str,
        *,
        router_id: str | None = None,
        intent: RoutingIntent = RoutingIntent.IDLE,
    ) -> RoutingDecision:
        return cls(capability_id=None, intent=intent, reason=reason, router_id=router_id)

    @classmethod
    def finish(cls, reason: str, *, router_id: str | None = None) -> RoutingDecision:
        return cls.no_action(reason, router_id=router_id, intent=RoutingIntent.FINISH)

    @classmethod
    def ask_human(cls, reason: str, *, router_id: str | None = None) -> RoutingDecision:
        return cls.no_action(reason, router_id=router_id, intent=RoutingIntent.ASK_HUMAN)


class ModelUsage(FrozenModel):
    """Resource usage of a model-backed routing call.

    Every field is optional: ``None`` means *unknown*, never zero. Adapters
    report only what their provider actually returned.
    """

    model_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class RoutingAttempt(FrozenModel):
    """One router's attempt at producing a decision (successful or not)."""

    router_id: str
    router_type: str
    success: bool
    decision: RoutingDecision | None = None  # also set when a parsed decision failed validation
    error: ErrorInfo | None = None
    latency_s: float = Field(default=0.0, ge=0.0)
    model: str | None = None  # model the provider reports as having served the call
    usage: ModelUsage | None = None
    request: dict[str, Any] | None = None  # serialised routing request, when one was built
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingOutcome(FrozenModel):
    """A decision plus the attempts that produced it (several when fallback ran)."""

    decision: RoutingDecision
    attempts: tuple[RoutingAttempt, ...] = ()

    @property
    def fallback_used(self) -> bool:
        return any(not a.success for a in self.attempts)
