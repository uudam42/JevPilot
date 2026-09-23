from domains.demo import ArithmeticDomain, NumberState, routing_rules
from jevpilot import (
    RuleRouter,
    Runtime,
    ScriptedRouter,
    TraceEventType,
    WorkflowStatus,
)


def _run(router):  # type: ignore[no-untyped-def]
    rt = Runtime()
    domain = rt.load(ArithmeticDomain())
    assert isinstance(domain, ArithmeticDomain)
    return rt.controller(router).run(domain.new_workflow(start=3, target=20))


def test_end_to_end_with_rule_router() -> None:
    result = _run(RuleRouter(routing_rules()))
    s = result.state
    assert isinstance(s, NumberState)
    assert result.succeeded and s.status is WorkflowStatus.SUCCEEDED
    assert s.value == 20
    assert [h.decision.capability_id for h in s.history] == [
        "arith.multiply",
        "arith.multiply",
        "arith.add",
        "arith.compare",
    ]
    assert len(s.candidate_solutions) == 1 and s.candidate_solutions[0].provenance


def test_router_is_replaceable_same_outcome() -> None:
    scripted = ScriptedRouter(
        [
            ("arith.multiply", {"factor": 2}),
            ("arith.multiply", {"factor": 2}),
            ("arith.add", {"amount": 8}),
            ("arith.compare", {}),
        ]
    )
    a = _run(scripted).state
    b = _run(RuleRouter(routing_rules())).state
    assert isinstance(a, NumberState) and isinstance(b, NumberState)
    assert a.status is b.status is WorkflowStatus.SUCCEEDED and a.value == b.value


def test_trace_reconstructs_every_step() -> None:
    result = _run(RuleRouter(routing_rules()))
    types = [e.type for e in result.trace]
    assert types[0] is TraceEventType.WORKFLOW_STARTED
    assert types[-1] is TraceEventType.WORKFLOW_COMPLETED
    # Each executed step: decision → observation → snapshot → evaluation → control
    per_step = [
        TraceEventType.ROUTING_DECISION,
        TraceEventType.OBSERVATION,
        TraceEventType.STATE_SNAPSHOT,
        TraceEventType.EVALUATION,
        TraceEventType.CONTROL_DECISION,
    ]
    body = types[2:-1]
    assert body == per_step * result.state.step
    assert [e.seq for e in result.trace] == list(range(len(result.trace)))
    # every observation's provenance is retained in state
    ids = {p.id for p in result.state.provenance}
    assert all(o.provenance.id in ids for o in result.state.observations)
