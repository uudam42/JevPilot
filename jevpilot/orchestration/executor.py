"""Executor: resolves, validates and invokes the capability a router selected.

    RoutingDecision → resolve → validate inputs → execute → validate output → Observation

The executor never chooses what to run and never raises for capability
failures; every outcome becomes a structured :class:`Observation`.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ValidationError

from jevpilot.core import (
    ErrorInfo,
    Observation,
    Provenance,
    RoutingDecision,
    WorkflowState,
    new_id,
    stable_digest,
    to_jsonable,
    utcnow,
)
from jevpilot.exceptions import CapabilityError, CapabilityNotFoundError
from jevpilot.interfaces.capability import (
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ExecutionContext,
)
from jevpilot.registry.capability_registry import CapabilityRegistry


class Executor:
    executor_id = "jevpilot.executor"

    def __init__(self, registry: CapabilityRegistry, *, check_preconditions: bool = True) -> None:
        self._registry = registry
        self._check_preconditions = check_preconditions

    def execute(self, decision: RoutingDecision, state: WorkflowState) -> Observation:
        if decision.capability_id is None:
            raise ValueError("cannot execute a no-action routing decision")
        execution_id = new_id("exec")
        started = utcnow()

        def fail(
            error: ErrorInfo, spec: CapabilitySpec | None = None, elapsed: float = 0.0
        ) -> Observation:
            prov = self._provenance(
                state, decision, execution_id, spec, decision.inputs, started, CapabilityResult()
            )
            return Observation(
                workflow_id=state.workflow_id,
                step=state.step,
                capability_id=decision.capability_id,
                decision_id=decision.id,
                success=False,
                error=error,
                provenance=prov,
                execution_time=elapsed,
            )

        # 1. resolve
        try:
            capability = self._registry.get(decision.capability_id)
        except CapabilityNotFoundError:
            return fail(
                ErrorInfo(
                    type="CapabilityNotFound",
                    message=f"no capability {decision.capability_id!r} is registered",
                    retryable=False,
                )
            )
        spec = capability.spec

        # 2. preconditions
        if self._check_preconditions and not _safe_applicable(capability, state):
            return fail(
                ErrorInfo(
                    type="PreconditionFailed",
                    message=f"{spec.id} is not applicable in the current state",
                    retryable=False,
                ),
                spec,
            )

        # 3. validate inputs
        try:
            inputs = _validate(spec.input_schema, decision.inputs)
        except ValidationError as exc:
            return fail(
                ErrorInfo(
                    type="InputValidationError",
                    message=str(exc),
                    retryable=False,
                    details={"errors": to_jsonable(exc.errors())},
                ),
                spec,
            )

        # 4. execute
        ctx = ExecutionContext(
            workflow_id=state.workflow_id,
            step=state.step,
            execution_id=execution_id,
            decision_id=decision.id,
            state=state,
        )
        t0 = time.perf_counter()
        try:
            raw = capability.execute(inputs, ctx)
        except CapabilityError as exc:
            return fail(
                ErrorInfo(type=type(exc).__name__, message=str(exc), retryable=exc.retryable),
                spec,
                time.perf_counter() - t0,
            )
        except Exception as exc:  # capability bugs become observations, not crashes
            return fail(
                ErrorInfo(type=type(exc).__name__, message=str(exc), retryable=True),
                spec,
                time.perf_counter() - t0,
            )
        elapsed = time.perf_counter() - t0
        result = raw if isinstance(raw, CapabilityResult) else CapabilityResult(output=raw)

        # 5. validate output
        try:
            output = _validate(spec.output_schema, result.output)
        except ValidationError as exc:
            return fail(
                ErrorInfo(
                    type="OutputValidationError",
                    message=str(exc),
                    retryable=False,
                    details={"errors": to_jsonable(exc.errors())},
                ),
                spec,
                elapsed,
            )

        # 6. observation with provenance stamped on everything produced
        prov = self._provenance(state, decision, execution_id, spec, _dump(inputs), started, result)
        artifacts = tuple(
            a if a.provenance else a.model_copy(update={"provenance": prov})
            for a in result.artifacts
        )
        candidates = tuple(
            c if c.provenance else c.model_copy(update={"provenance": prov})
            for c in result.effects.candidates
        )
        effects = result.effects.model_copy(update={"candidates": candidates})
        return Observation(
            workflow_id=state.workflow_id,
            step=state.step,
            capability_id=spec.id,
            decision_id=decision.id,
            success=True,
            result=output,
            uncertainty=result.uncertainty,
            artifacts=artifacts,
            effects=effects,
            provenance=prov,
            execution_time=elapsed,
            metadata=result.metadata,
        )

    def _provenance(
        self,
        state: WorkflowState,
        decision: RoutingDecision,
        execution_id: str,
        spec: CapabilitySpec | None,
        inputs: dict[str, Any],
        started: Any,
        result: CapabilityResult,
    ) -> Provenance:
        return Provenance(
            workflow_id=state.workflow_id,
            step=state.step,
            execution_id=execution_id,
            capability_id=decision.capability_id,
            capability_version=spec.version if spec else None,
            domain=spec.domain if spec else None,
            inputs=to_jsonable(inputs),
            inputs_digest=stable_digest(inputs),
            sources=result.sources,
            derived_from=result.derived_from,
            started_at=started,
            finished_at=utcnow(),
            metadata={"executor": self.executor_id, "router_id": decision.router_id},
        )


def _validate(schema: type[BaseModel] | None, value: Any) -> Any:
    if schema is None or isinstance(value, schema):
        return value
    return schema.model_validate(value)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {"value": to_jsonable(value)}


def _safe_applicable(capability: Capability, state: WorkflowState) -> bool:
    try:
        return capability.is_applicable(state)
    except Exception:
        return False
