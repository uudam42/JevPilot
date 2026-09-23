"""Router interface: π(S_t, available capabilities) → RoutingDecision.

The contract every routing policy satisfies, whether it is rule-based,
scripted, learned, LLM-backed or Jev-backed:

- input: the workflow state and the :class:`CapabilitySpec` descriptors of the
  capabilities currently available (never executable objects);
- output: a :class:`RoutingDecision`, which must name an offered capability and
  satisfy its input schema (:func:`validate_decision`), or propose no action.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from jevpilot.core import (
    ErrorInfo,
    RoutingAttempt,
    RoutingDecision,
    RoutingOutcome,
    WorkflowState,
    to_jsonable,
)
from jevpilot.exceptions import (
    InvalidCapabilityError,
    InvalidRoutingInputError,
    MalformedRoutingDecisionError,
    RoutingError,
)
from jevpilot.interfaces.capability import CapabilitySpec


class Router(ABC):
    """Chooses the next capability. Routers decide; they never execute.

    The state already carries the goal, plan, history of previous actions and
    past evaluations, so ``(state, capabilities)`` is the complete routing
    input for rule-based, learned, LLM-based or Jev-based policies alike.

    Implementations signal an explicit routing failure by raising a
    :class:`~jevpilot.exceptions.RoutingError` subclass.
    """

    router_id: str = "router"

    @abstractmethod
    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision: ...

    def route(self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]) -> RoutingOutcome:
        """The decision plus diagnostics. The controller calls this method.

        The default wraps :meth:`select_next` in a single timed attempt.
        Model-backed and composite routers override it to report the routing
        request, model usage and fallback attempts.
        """
        t0 = time.perf_counter()
        decision = self.select_next(state, capabilities)
        latency = time.perf_counter() - t0
        if not isinstance(decision, RoutingDecision):
            message = f"router returned {type(decision).__name__}, not RoutingDecision"
            error = ErrorInfo(
                type="MalformedRoutingDecisionError", message=message, retryable=False
            )
            raise MalformedRoutingDecisionError(
                message, attempts=(self.attempt(success=False, error=error, latency_s=latency),)
            )
        attempt = self.attempt(success=True, decision=decision, latency_s=latency)
        return RoutingOutcome(decision=decision, attempts=(attempt,))

    def config(self) -> dict[str, Any]:
        """JSON-serialisable configuration, recorded for reproducibility."""
        return {"router_id": self.router_id, "type": type(self).__name__}

    def attempt(self, **fields: Any) -> RoutingAttempt:
        """A :class:`RoutingAttempt` attributed to this router."""
        return RoutingAttempt(router_id=self.router_id, router_type=type(self).__name__, **fields)


def validate_decision(decision: RoutingDecision, capabilities: Sequence[CapabilitySpec]) -> None:
    """Reject a decision that could not be executed as stated.

    Raises :class:`InvalidCapabilityError` when the capability was not offered,
    and :class:`InvalidRoutingInputError` when the inputs are not plain JSON
    data or do not satisfy the capability's input schema. A no-action decision
    is always valid.
    """
    if decision.capability_id is None:
        return
    offered = {c.id: c for c in capabilities}
    spec = offered.get(decision.capability_id)
    if spec is None:
        raise InvalidCapabilityError(
            f"capability {decision.capability_id!r} is not among the available capabilities",
            details={"capability_id": decision.capability_id, "available": sorted(offered)},
        )
    problem = json_data_problem(decision.inputs)
    if problem is not None:
        raise InvalidRoutingInputError(
            f"inputs for {spec.id} are not plain JSON data: {problem}",
            details={"capability_id": spec.id},
        )
    if spec.input_schema is not None:
        try:
            spec.input_schema.model_validate(decision.inputs)
        except ValidationError as exc:
            raise InvalidRoutingInputError(
                f"inputs for {spec.id} do not match its input schema: {exc.error_count()} error(s)",
                details={"capability_id": spec.id, "errors": to_jsonable(exc.errors())},
            ) from None


def json_data_problem(value: Any, path: str = "inputs") -> str | None:
    """Describe why ``value`` is not plain JSON data (``None`` if it is).

    Decisions must be data, never objects: this keeps model output from
    smuggling executable or stateful objects into capability inputs.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} has a non-string key {key!r}"
            problem = json_data_problem(item, f"{path}.{key}")
            if problem:
                return problem
        return None
    if isinstance(value, list | tuple):
        for i, item in enumerate(value):
            problem = json_data_problem(item, f"{path}[{i}]")
            if problem:
                return problem
        return None
    return f"{path} is a {type(value).__name__}"


def error_info(exc: RoutingError) -> ErrorInfo:
    """Structured description of a routing failure."""
    return ErrorInfo(
        type=type(exc).__name__, message=str(exc), retryable=exc.retryable, details=exc.details
    )
