from jevpilot import (
    ControlAction,
    ControlDecision,
    DefaultControlPolicy,
    ErrorInfo,
    EvaluationResult,
    HistoryEntry,
    Observation,
    Provenance,
    RoutingDecision,
    WorkflowState,
)

A = ControlAction


def _ev(rec: ControlAction) -> EvaluationResult:
    return EvaluationResult(evaluator_id="e", recommendation=rec)


def _fail(state: WorkflowState, retryable: bool = True) -> Observation:
    return Observation(
        workflow_id=state.workflow_id,
        step=0,
        capability_id="c",
        success=False,
        error=ErrorInfo(type="E", message="m", retryable=retryable),
        provenance=Provenance(),
    )


def _history(*actions: ControlAction) -> tuple[HistoryEntry, ...]:
    d = RoutingDecision(capability_id="c")
    return tuple(
        HistoryEntry(step=i, decision=d, control=ControlDecision(action=a))
        for i, a in enumerate(actions)
    )


def test_follows_terminal_recommendations(state: WorkflowState) -> None:
    p = DefaultControlPolicy()
    for rec in (A.TERMINATE_SUCCESS, A.TERMINATE_FAILURE, A.HUMAN_INTERVENTION):
        assert p.decide(state, _ev(rec), None).action is rec


def test_step_budget(state: WorkflowState) -> None:
    p = DefaultControlPolicy(max_steps=3)
    assert p.decide(state.evolve(step=3), _ev(A.CONTINUE), None).action is A.TERMINATE_FAILURE


def test_retry_then_give_up_retrying(state: WorkflowState) -> None:
    p = DefaultControlPolicy(max_retries=2)
    obs = _fail(state)
    s = state.evolve(observations=(obs,))
    assert p.decide(s, _ev(A.CONTINUE), obs).action is A.RETRY
    s2 = s.evolve(history=_history(A.RETRY, A.RETRY))
    assert p.decide(s2, _ev(A.CONTINUE), obs).action is A.CONTINUE


def test_non_retryable_failure_continues(state: WorkflowState) -> None:
    obs = _fail(state, retryable=False)
    s = state.evolve(observations=(obs,))
    assert DefaultControlPolicy().decide(s, _ev(A.CONTINUE), obs).action is A.CONTINUE


def test_consecutive_failure_cap(state: WorkflowState) -> None:
    obs = _fail(state)
    s = state.evolve(observations=(obs,) * 3)
    p = DefaultControlPolicy(max_consecutive_failures=3)
    assert p.decide(s, _ev(A.CONTINUE), obs).action is A.TERMINATE_FAILURE


def test_replan_budget(state: WorkflowState) -> None:
    p = DefaultControlPolicy(max_replans=1)
    assert p.decide(state, _ev(A.REPLAN), None).action is A.REPLAN
    s = state.evolve(history=_history(A.REPLAN))
    assert p.decide(s, _ev(A.REPLAN), None).action is A.HUMAN_INTERVENTION
