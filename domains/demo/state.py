"""Domain-extended state: shows how a domain adds typed fields to WorkflowState."""

from __future__ import annotations

from jevpilot import WorkflowState


class NumberState(WorkflowState):
    value: float = 0.0
    target: float = 0.0
