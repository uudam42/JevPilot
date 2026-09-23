"""Minimal planners. Real planners (LLM, Jev, rule-based) plug in via the Planner interface."""

from __future__ import annotations

from collections.abc import Sequence

from jevpilot.core import Plan, PlanStep, WorkflowState
from jevpilot.interfaces.planner import Planner


class NullPlanner(Planner):
    """Produces an empty plan: the router alone drives execution."""

    planner_id = "null_planner"

    def plan(self, state: WorkflowState) -> Plan:
        return Plan(rationale="no planning", planner_id=self.planner_id)


class StaticPlanner(Planner):
    """Returns a fixed sequence of steps regardless of state."""

    planner_id = "static_planner"

    def __init__(self, steps: Sequence[PlanStep], rationale: str = "") -> None:
        self._steps = tuple(steps)
        self._rationale = rationale

    def plan(self, state: WorkflowState) -> Plan:
        return Plan(steps=self._steps, rationale=self._rationale, planner_id=self.planner_id)
