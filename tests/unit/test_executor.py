from typing import Any

from pydantic import BaseModel

from jevpilot import (
    Artifact,
    Candidate,
    CapabilityRegistry,
    CapabilityResult,
    Executor,
    RoutingDecision,
    SourceRef,
    StateEffects,
    WorkflowState,
)
from jevpilot.exceptions import CapabilityError
from tests.conftest import EchoInput, make_capability


class Out(BaseModel):
    n: int


def _decide(cid: str, **inputs: Any) -> RoutingDecision:
    return RoutingDecision(capability_id=cid, inputs=inputs, router_id="test")


def test_success_produces_structured_observation(
    registry: CapabilityRegistry, state: WorkflowState
) -> None:
    registry.register(
        make_capability(
            "t.echo",
            lambda i, ctx: i.text.upper(),
            input_schema=EchoInput,
            domain="t",
            version="1.2",
        )
    )
    obs = Executor(registry).execute(_decide("t.echo", text="hi"), state)
    assert obs.success and obs.result == "HI" and obs.error is None
    p = obs.provenance
    assert p.capability_id == "t.echo" and p.capability_version == "1.2" and p.domain == "t"
    assert p.inputs == {"text": "hi"} and p.inputs_digest
    assert p.workflow_id == state.workflow_id and p.step == state.step
    assert obs.execution_time >= 0


def test_unknown_capability_is_failed_observation(
    registry: CapabilityRegistry, state: WorkflowState
) -> None:
    obs = Executor(registry).execute(_decide("missing"), state)
    assert not obs.success and obs.error and obs.error.type == "CapabilityNotFound"
    assert not obs.error.retryable


def test_input_validation(registry: CapabilityRegistry, state: WorkflowState) -> None:
    registry.register(make_capability("t.echo", input_schema=EchoInput))
    obs = Executor(registry).execute(_decide("t.echo", wrong=1), state)
    assert not obs.success and obs.error and obs.error.type == "InputValidationError"


def test_output_validation(registry: CapabilityRegistry, state: WorkflowState) -> None:
    registry.register(make_capability("t.good", lambda i, c: {"n": 1}, output_schema=Out))
    registry.register(make_capability("t.bad", lambda i, c: {"n": "x"}, output_schema=Out))
    ex = Executor(registry)
    assert ex.execute(_decide("t.good"), state).result == Out(n=1)
    bad = ex.execute(_decide("t.bad"), state)
    assert not bad.success and bad.error and bad.error.type == "OutputValidationError"


def test_exceptions_become_observations(registry: CapabilityRegistry, state: WorkflowState) -> None:
    def crash(i: Any, c: Any) -> None:
        raise ZeroDivisionError("oops")

    def controlled(i: Any, c: Any) -> None:
        raise CapabilityError("permanent", retryable=False)

    registry.register(make_capability("t.crash", crash))
    registry.register(make_capability("t.ctl", controlled))
    ex = Executor(registry)
    o1 = ex.execute(_decide("t.crash"), state)
    assert o1.error and o1.error.type == "ZeroDivisionError" and o1.error.retryable
    o2 = ex.execute(_decide("t.ctl"), state)
    assert o2.error and not o2.error.retryable


def test_precondition_enforced(registry: CapabilityRegistry, state: WorkflowState) -> None:
    registry.register(make_capability("t.never", applicable=lambda s: False))
    obs = Executor(registry).execute(_decide("t.never"), state)
    assert obs.error and obs.error.type == "PreconditionFailed"


def test_provenance_stamped_on_artifacts_and_candidates(
    registry: CapabilityRegistry, state: WorkflowState
) -> None:
    def produce(i: Any, c: Any) -> CapabilityResult:
        return CapabilityResult(
            artifacts=(Artifact(name="a"),),
            effects=StateEffects(candidates=(Candidate(content=1),)),
            sources=(SourceRef(kind="database", identifier="db1"),),
            derived_from=("prov_upstream",),
        )

    registry.register(make_capability("t.produce", produce))
    obs = Executor(registry).execute(_decide("t.produce"), state)
    assert obs.artifacts[0].provenance == obs.provenance
    assert obs.effects.candidates[0].provenance == obs.provenance
    assert obs.provenance.sources[0].identifier == "db1"
    assert obs.provenance.derived_from == ("prov_upstream",)
