"""State reducers: domain-specific interpretation of observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jevpilot.core import Observation, WorkflowState


class StateReducer(ABC):
    """Pure function ``(S, O) → S'`` applied after the generic state update.

    Used by domains whose state subclass has fields the core cannot know about.
    Reducers must return a new state and must not mutate their inputs.
    """

    @abstractmethod
    def reduce(self, state: WorkflowState, observation: Observation) -> WorkflowState: ...
