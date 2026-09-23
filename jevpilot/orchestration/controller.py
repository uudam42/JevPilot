"""The JevPilot control loop.

    while not terminal:
        capabilities = registry.available(state)
        decision     = router.route(state, capabilities).decision  # decide (validated)
        observation  = executor.execute(decision, state)           # act
        state        = state_manager.update(state, decision, obs)  # S_{t+1}
        evaluation   = evaluator.evaluate(state)                   # assess
        control      = policy.decide(state, evaluation, obs)       # continue/retry/replan/stop

The controller contains no domain knowledge and no termination rules of its
own beyond a hard iteration safety cap; both routing and termination are
injected.

A router that raises :class:`~jevpilot.exceptions.RoutingError`, or returns a
decision that fails :func:`~jevpilot.interfaces.router.validate_decision`,
produces an explicit routing failure: it is traced, recorded in history and
handed to ``policy.on_routing_failure``. Nothing is executed for it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from jevpilot.core import (
    ControlAction,
    ControlDecision,
    ErrorInfo,
    EvaluationResult,
    Observation,
    RoutingDecision,
    RoutingIntent,
    TraceEvent,
    TraceEventType,
    Tracer,
    TraceSink,
    WorkflowState,
    WorkflowStatus,
)
from jevpilot.core.trace import InMemoryTraceSink
from jevpilot.exceptions import RoutingError
from jevpilot.interfaces.evaluator import Evaluator
from jevpilot.interfaces.planner import Planner
from jevpilot.interfaces.policy import ControlPolicy
from jevpilot.interfaces.router import Router, error_info, validate_decision
from jevpilot.orchestration.executor import Executor
from jevpilot.orchestration.policies import DefaultControlPolicy
from jevpilot.orchestration.state_manager import StateManager
from jevpilot.registry.capability_registry import CapabilityRegistry

_FINAL_STATUS = {
    ControlAction.TERMINATE_SUCCESS: WorkflowStatus.SUCCEEDED,
    ControlAction.TERMINATE_FAILURE: WorkflowStatus.FAILED,
    ControlAction.HUMAN_INTERVENTION: WorkflowStatus.AWAITING_HUMAN,
}


@dataclass(frozen=True)
class WorkflowResult:
    state: WorkflowState
    control: ControlDecision
    trace: tuple[TraceEvent, ...]
    error: ErrorInfo | None = None

    @property
    def succeeded(self) -> bool:
        return self.state.status is WorkflowStatus.SUCCEEDED


class Controller:
    def __init__(
        self,
        registry: CapabilityRegistry,
        router: Router,
        evaluator: Evaluator,
        *,
        executor: Executor | None = None,
        planner: Planner | None = None,
        policy: ControlPolicy | None = None,
        state_manager: StateManager | None = None,
        trace_sinks: Sequence[TraceSink] = (),
        max_iterations: int = 1000,
        snapshot_states: bool = True,
    ) -> None:
        self.registry = registry
        self.router = router
        self.evaluator = evaluator
        self.executor = executor or Executor(registry)
        self.planner = planner
        self.policy = policy or DefaultControlPolicy()
        self.state_manager = state_manager or StateManager()
        self.trace_sinks = tuple(trace_sinks)
        self.max_iterations = max_iterations
        self.snapshot_states = snapshot_states

    def run(self, state: WorkflowState) -> WorkflowResult:
        recorder = InMemoryTraceSink()
        tracer = Tracer([recorder, *self.trace_sinks])
        run = _Run(self, tracer)
        try:
            final, control = run.execute(state)
            error = None
        except Exception as exc:
            error = ErrorInfo(type=type(exc).__name__, message=str(exc), retryable=False)
            final = self.state_manager.set_status(run.state, WorkflowStatus.FAILED)
            control = ControlDecision(
                action=ControlAction.TERMINATE_FAILURE,
                reason=f"orchestration error: {error.message}",
                source="controller",
            )
            tracer.emit(final.workflow_id, final.step, TraceEventType.ERROR, error=error)
        tracer.emit(
            final.workflow_id,
            final.step,
            TraceEventType.WORKFLOW_COMPLETED,
            status=final.status,
            control=control,
        )
        return WorkflowResult(final, control, tuple(recorder.events), error)


class _Run:
    """State of a single :meth:`Controller.run` invocation."""

    def __init__(self, ctl: Controller, tracer: Tracer) -> None:
        self.ctl = ctl
        self.tracer = tracer
        self.state: WorkflowState  # latest state, kept for error reporting

    def emit(self, type: TraceEventType, **payload: object) -> None:
        self.tracer.emit(self.state.workflow_id, self.state.step, type, **payload)

    def snapshot(self) -> None:
        if self.ctl.snapshot_states:
            self.emit(TraceEventType.STATE_SNAPSHOT, state=self.state)

    def execute(self, initial: WorkflowState) -> tuple[WorkflowState, ControlDecision]:
        ctl, sm = self.ctl, self.ctl.state_manager
        self.state = sm.start(initial)
        self.emit(
            TraceEventType.WORKFLOW_STARTED,
            goal=self.state.goal,
            router=ctl.router.router_id,
            router_config=ctl.router.config(),
            evaluator=ctl.evaluator.evaluator_id,
            policy=ctl.policy.policy_id,
        )
        self.snapshot()
        self._plan()

        retry: RoutingDecision | None = None
        for _ in range(ctl.max_iterations):
            observation: Observation | None = None
            try:
                decision: RoutingDecision | None = self._route(retry)
            except RoutingError as exc:
                decision = None
                control = self._routing_failed(exc)
            if decision is None:
                pass
            elif decision.is_no_action:
                self.state = sm.record_no_action(self.state, decision)
                evaluation = self._evaluate()
                if evaluation.recommendation.is_terminal:
                    control = ControlDecision(
                        action=evaluation.recommendation,
                        reason=f"router proposed no action ({decision.intent}); "
                        f"evaluator: {evaluation.rationale}",
                        source="controller",
                    )
                elif decision.intent is RoutingIntent.ASK_HUMAN:
                    control = ControlDecision(
                        action=ControlAction.HUMAN_INTERVENTION,
                        reason=f"router requests human input: {decision.reason}",
                        source="controller",
                    )
                else:
                    control = ControlDecision(
                        action=ControlAction.HUMAN_INTERVENTION,
                        reason=f"router proposed no action ({decision.intent}: {decision.reason}) "
                        "and the goal is not met",
                        source="controller",
                    )
            else:
                observation = ctl.executor.execute(decision, self.state)
                self.emit(TraceEventType.OBSERVATION, observation=observation)
                self.state = sm.update(self.state, decision, observation)
                self.snapshot()
                evaluation = self._evaluate()
                control = ctl.policy.decide(self.state, evaluation, observation)

            self.state = sm.record_control(self.state, control)
            self.emit(TraceEventType.CONTROL_DECISION, control=control)

            retry = None
            if control.action in _FINAL_STATUS:
                self.state = sm.set_status(self.state, _FINAL_STATUS[control.action])
                return self.state, control
            if (
                control.action is ControlAction.RETRY
                and decision is not None
                and not decision.is_no_action
            ):
                retry = decision
            elif control.action is ControlAction.REPLAN:
                self._plan()

        control = ControlDecision(
            action=ControlAction.TERMINATE_FAILURE,
            reason=f"controller safety cap of {ctl.max_iterations} iterations reached",
            source="controller",
        )
        self.state = sm.set_status(self.state, WorkflowStatus.FAILED)
        return self.state, control

    def _route(self, retry: RoutingDecision | None) -> RoutingDecision:
        """Obtain and validate the next decision; emits ``routing_decision``.

        Raises :class:`RoutingError` (after emitting ``routing_failure``) when
        the router fails or its decision is invalid.
        """
        if retry is not None:
            decision = RoutingDecision(
                capability_id=retry.capability_id,
                inputs=retry.inputs,
                reason=f"retry of {retry.id}",
                confidence=retry.confidence,
                router_id=retry.router_id,
                metadata={**retry.metadata, "retry_of": retry.id},
            )
            self.emit(TraceEventType.ROUTING_DECISION, decision=decision, routing_latency_s=0.0)
            return decision
        capabilities = self.ctl.registry.available(self.state)
        if not capabilities:
            decision = RoutingDecision.no_action(
                "no capabilities available", router_id="controller"
            )
            self.emit(TraceEventType.ROUTING_DECISION, decision=decision, routing_latency_s=0.0)
            return decision
        available = [c.id for c in capabilities]
        t0 = time.perf_counter()
        try:
            outcome = self.ctl.router.route(self.state, capabilities)
            # Defence in depth: whatever the router did internally, nothing that
            # was not offered, or does not match its schema, reaches the executor.
            validate_decision(outcome.decision, capabilities)
        except RoutingError as exc:
            self.emit(
                TraceEventType.ROUTING_FAILURE,
                error=error_info(exc),
                routing_latency_s=time.perf_counter() - t0,
                available_capabilities=available,
                attempts=exc.attempts,
            )
            raise
        self.emit(
            TraceEventType.ROUTING_DECISION,
            decision=outcome.decision,
            routing_latency_s=time.perf_counter() - t0,
            available_capabilities=available,
            fallback_used=outcome.fallback_used,
            attempts=outcome.attempts,
        )
        return outcome.decision

    def _routing_failed(self, exc: RoutingError) -> ControlDecision:
        error = error_info(exc)
        placeholder = RoutingDecision.no_action(
            f"routing failed: {error.type}", router_id=self.ctl.router.router_id
        )
        self.state = self.ctl.state_manager.record_routing_failure(self.state, placeholder, error)
        return self.ctl.policy.on_routing_failure(self.state, error)

    def _evaluate(self) -> EvaluationResult:
        evaluation = self.ctl.evaluator.evaluate(self.state)
        self.state = self.ctl.state_manager.apply_evaluation(self.state, evaluation)
        self.emit(TraceEventType.EVALUATION, evaluation=evaluation)
        return evaluation

    def _plan(self) -> None:
        if self.ctl.planner is None:
            return
        plan = self.ctl.planner.plan(self.state)
        self.state = self.ctl.state_manager.set_plan(self.state, plan)
        self.emit(TraceEventType.PLAN_CREATED, plan=self.state.plan)
