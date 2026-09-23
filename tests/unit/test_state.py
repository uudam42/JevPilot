import pytest
from pydantic import ValidationError

from jevpilot import Goal, Uncertainty, WorkflowState, WorkflowStatus


def test_state_is_immutable(state: WorkflowState) -> None:
    with pytest.raises(ValidationError):
        state.step = 5  # type: ignore[misc]


def test_evolve_returns_new_instance(state: WorkflowState) -> None:
    new = state.evolve(step=3)
    assert new.step == 3 and state.step == 0
    assert new.workflow_id == state.workflow_id


def test_domain_state_subclass_survives_evolve() -> None:
    class DomainState(WorkflowState):
        extra: int = 1

    s = DomainState(goal=Goal(description="g"))
    s2 = s.evolve(extra=2, step=1)
    assert isinstance(s2, DomainState) and s2.extra == 2


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkflowState(goal=Goal(description="g"), unknown_field=1)  # type: ignore[call-arg]


def test_status_finality() -> None:
    assert not WorkflowStatus.RUNNING.is_final
    assert WorkflowStatus.AWAITING_HUMAN.is_final
    assert WorkflowStatus.SUCCEEDED.is_final


def test_uncertainty_bounds() -> None:
    assert Uncertainty.certain().confidence == 1.0
    with pytest.raises(ValidationError):
        Uncertainty(confidence=1.5)


def test_state_serialises_to_json(state: WorkflowState) -> None:
    assert WorkflowState.model_validate_json(state.model_dump_json()) == state
