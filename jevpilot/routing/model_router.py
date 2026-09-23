"""Shared pipeline for routers backed by an external model.

    WorkflowState + CapabilitySpecs
        → RoutingRequest            (canonical JSON problem)
        → _infer()                  (subclass: LLM prompt + adapter, Jev adapter, …)
        → RoutingModelResponse      (raw, untrusted)
        → decision_from_output()    (strict parse)
        → validate_decision()       (offered capability, schema-valid inputs)
        → RoutingDecision

Every failure along the way becomes a :class:`~jevpilot.exceptions.RoutingError`
carrying a :class:`~jevpilot.core.RoutingAttempt` (request, latency, usage,
error). Nothing falls back silently here. Fallback is a separate, explicit
router (:class:`~jevpilot.routing.fallback.FallbackRouter`).
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from jevpilot.core import RoutingDecision, RoutingOutcome, WorkflowState
from jevpilot.exceptions import (
    LowConfidenceRoutingError,
    MalformedRoutingDecisionError,
    RouterAdapterError,
    RouterTimeoutError,
    RoutingError,
)
from jevpilot.interfaces.capability import CapabilitySpec
from jevpilot.interfaces.router import Router, error_info, validate_decision
from jevpilot.routing.contracts import RoutingModelResponse
from jevpilot.routing.parsing import decision_from_output
from jevpilot.routing.request import RoutingRequest


class ModelRouter(Router):
    """Base class for adapter-backed routers (LLM, Jev, future learned models).

    ``timeout_s`` is a soft deadline: it is passed to the adapter, and a
    response arriving later is discarded as :class:`RouterTimeoutError`.
    ``min_confidence`` (off by default) rejects decisions whose self-reported
    confidence is lower or missing. Confidence is never treated as calibrated.
    """

    router_id = "model_router"

    def __init__(
        self,
        *,
        router_id: str | None = None,
        timeout_s: float | None = None,
        min_confidence: float | None = None,
        max_actions: int = 20,
        trace_requests: bool = True,
    ) -> None:
        if router_id is not None:
            self.router_id = router_id
        self.timeout_s = timeout_s
        self.min_confidence = min_confidence
        self.max_actions = max_actions
        self.trace_requests = trace_requests

    @abstractmethod
    def _infer(self, request: RoutingRequest) -> RoutingModelResponse:
        """Ask the model. May raise; non-routing exceptions become adapter errors."""

    @abstractmethod
    def adapter_config(self) -> dict[str, Any]:
        """Adapter/model configuration recorded with every run."""

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "timeout_s": self.timeout_s,
            "min_confidence": self.min_confidence,
            "max_actions": self.max_actions,
            "adapter": self.adapter_config(),
        }

    def build_request(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingRequest:
        return RoutingRequest.from_state(
            state, capabilities, max_actions=self.max_actions, metadata={"router": self.router_id}
        )

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        return self.route(state, capabilities).decision

    def route(self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]) -> RoutingOutcome:
        request = self.build_request(state, capabilities)
        record = request.to_dict() if self.trace_requests else None
        response: RoutingModelResponse | None = None
        decision: RoutingDecision | None = None
        t0 = time.perf_counter()

        def record_failure(exc: RoutingError) -> None:
            exc.attempts = (
                *exc.attempts,
                self.attempt(
                    success=False,
                    decision=decision,
                    error=error_info(exc),
                    latency_s=time.perf_counter() - t0,
                    model=response.model if response else None,
                    usage=response.usage if response else None,
                    request=record,
                    metadata=dict(response.metadata) if response else {},
                ),
            )

        try:
            try:
                response = self._infer(request)
            except RoutingError:
                raise
            except TimeoutError as exc:
                raise RouterTimeoutError(f"model call timed out: {exc}") from exc
            except Exception as exc:  # provider/SDK failures of any kind
                raise RouterAdapterError(f"adapter failed: {type(exc).__name__}: {exc}") from exc
            if not isinstance(response, RoutingModelResponse):
                got = type(response).__name__
                response = None
                raise MalformedRoutingDecisionError(f"adapter returned {got}")
            latency = time.perf_counter() - t0
            if self.timeout_s is not None and latency > self.timeout_s:
                raise RouterTimeoutError(
                    f"model answered after {latency:.3f}s (deadline {self.timeout_s}s)"
                )
            decision = decision_from_output(response.output, router_id=self.router_id)
            validate_decision(decision, capabilities)
            if self.min_confidence is not None and (
                decision.confidence is None or decision.confidence < self.min_confidence
            ):
                raise LowConfidenceRoutingError(
                    f"confidence {decision.confidence} below threshold {self.min_confidence}",
                    details={"confidence": decision.confidence, "threshold": self.min_confidence},
                )
        except RoutingError as exc:
            record_failure(exc)
            raise

        attempt = self.attempt(
            success=True,
            decision=decision,
            latency_s=time.perf_counter() - t0,
            model=response.model,
            usage=response.usage,
            request=record,
            metadata=dict(response.metadata),
        )
        return RoutingOutcome(decision=decision, attempts=(attempt,))
