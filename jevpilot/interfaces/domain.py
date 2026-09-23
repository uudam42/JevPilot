"""Domain modules: the only way domain knowledge enters JevPilot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from jevpilot.core import Goal, WorkflowState
from jevpilot.interfaces.capability import Capability
from jevpilot.interfaces.evaluator import Evaluator
from jevpilot.interfaces.reducer import StateReducer


class DomainModule(ABC):
    """A pluggable bundle of capabilities, evaluators and state extensions.

    A domain depends on JevPilot's public interfaces; JevPilot never depends on
    a domain. Loading a domain requires no change to core code.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""

    @abstractmethod
    def capabilities(self) -> Sequence[Capability]:
        """Capabilities this domain contributes."""

    def evaluators(self) -> Sequence[Evaluator]:
        """Evaluators defining success for this domain's goals."""
        return ()

    def reducers(self) -> Sequence[StateReducer]:
        """Reducers for domain-specific state fields."""
        return ()

    def state_type(self) -> type[WorkflowState]:
        """State class used by this domain (override to extend the state)."""
        return WorkflowState

    def create_state(self, goal: Goal | str, **fields: Any) -> WorkflowState:
        """Build an initial state of :meth:`state_type` for ``goal``."""
        if isinstance(goal, str):
            goal = Goal(description=goal)
        metadata = {"domain": self.name, "domain_version": self.version}
        metadata.update(fields.pop("metadata", {}))
        return self.state_type()(goal=goal, metadata=metadata, **fields)
