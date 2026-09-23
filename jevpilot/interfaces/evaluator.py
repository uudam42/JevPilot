"""Evaluator interface. Domains decide what "good" means."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jevpilot.core import EvaluationResult, WorkflowState


class Evaluator(ABC):
    evaluator_id: str = "evaluator"

    @abstractmethod
    def evaluate(self, state: WorkflowState) -> EvaluationResult: ...
