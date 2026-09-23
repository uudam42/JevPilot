"""Convenience facade wiring registries, domains and a controller together."""

from __future__ import annotations

from collections.abc import Sequence

from jevpilot.core import TraceSink
from jevpilot.interfaces.domain import DomainModule
from jevpilot.interfaces.evaluator import Evaluator
from jevpilot.interfaces.planner import Planner
from jevpilot.interfaces.policy import ControlPolicy
from jevpilot.interfaces.router import Router
from jevpilot.orchestration.controller import Controller
from jevpilot.orchestration.evaluators import CompositeEvaluator, NullEvaluator
from jevpilot.orchestration.state_manager import StateManager
from jevpilot.registry.capability_registry import CapabilityRegistry
from jevpilot.registry.domain_registry import DomainRegistry


class Runtime:
    """One capability registry plus the domains loaded into it."""

    def __init__(self) -> None:
        self.capabilities = CapabilityRegistry()
        self.domains = DomainRegistry(self.capabilities)

    def load(self, domain: DomainModule | str) -> DomainModule:
        """Load a domain object, a ``"module:Attr"`` path, or an entry-point name."""
        if isinstance(domain, DomainModule):
            return self.domains.load(domain)
        if ":" in domain:
            return self.domains.load_from_path(domain)
        return self.domains.load_entry_point(domain)

    def controller(
        self,
        router: Router,
        *,
        domains: Sequence[str] | None = None,
        evaluator: Evaluator | None = None,
        planner: Planner | None = None,
        policy: ControlPolicy | None = None,
        trace_sinks: Sequence[TraceSink] = (),
        max_iterations: int = 1000,
    ) -> Controller:
        """Build a controller.

        ``domains`` scopes which loaded domains contribute evaluators and state
        reducers (default: all). All registered capabilities stay visible to
        the router, so cross-domain workflows remain possible.
        """
        scope = (
            [self.domains.get(n) for n in domains] if domains is not None else self.domains.list()
        )
        evaluators = [e for d in scope for e in d.evaluators()]
        reducers = [r for d in scope for r in d.reducers()]
        if evaluator is None:
            if not evaluators:
                evaluator = NullEvaluator()
            elif len(evaluators) == 1:
                evaluator = evaluators[0]
            else:
                evaluator = CompositeEvaluator(evaluators)
        return Controller(
            self.capabilities,
            router,
            evaluator,
            planner=planner,
            policy=policy,
            state_manager=StateManager(reducers),
            trace_sinks=trace_sinks,
            max_iterations=max_iterations,
        )
