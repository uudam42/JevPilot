"""Planner interface: goal/state → plan (subgoals, suggested steps)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jevpilot.core import Plan, WorkflowState


class Planner(ABC):
    """Produces an advisory :class:`Plan`. Independent of any router."""

    planner_id: str = "planner"

    @abstractmethod
    def plan(self, state: WorkflowState) -> Plan: ...
