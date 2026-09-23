"""Control-flow vocabulary shared by evaluators, policies and the controller."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jevpilot.core._base import FrozenModel


class ControlAction(StrEnum):
    """What the control loop should do next."""

    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    TERMINATE_SUCCESS = "terminate_success"
    TERMINATE_FAILURE = "terminate_failure"
    HUMAN_INTERVENTION = "human_intervention"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {
        ControlAction.TERMINATE_SUCCESS,
        ControlAction.TERMINATE_FAILURE,
        ControlAction.HUMAN_INTERVENTION,
    }
)


class ControlDecision(FrozenModel):
    """Final decision of a :class:`~jevpilot.interfaces.policy.ControlPolicy`."""

    action: ControlAction
    reason: str = ""
    source: str | None = None  # which policy/component made the decision
    metadata: dict[str, Any] = Field(default_factory=dict)
