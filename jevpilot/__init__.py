"""JevPilot — a domain-agnostic agentic orchestration framework.

Public API: import from here (or from the subpackages listed in ``__all__``).
Domains should depend only on names exported by this package.
"""

from jevpilot.core import *  # noqa: F403
from jevpilot.core import __all__ as _core_all
from jevpilot.interfaces import *  # noqa: F403
from jevpilot.interfaces import __all__ as _interfaces_all
from jevpilot.orchestration import *  # noqa: F403
from jevpilot.orchestration import __all__ as _orchestration_all
from jevpilot.planning import NullPlanner, StaticPlanner
from jevpilot.registry import CapabilityRegistry, DomainRegistry
from jevpilot.routing import (
    FallbackRouter,
    JevRouter,
    LLMRouter,
    RoutingRequest,
    Rule,
    RuleRouter,
    ScriptedRouter,
)

__version__ = "0.0.1"

__all__ = [
    *_core_all,
    *_interfaces_all,
    *_orchestration_all,
    "CapabilityRegistry",
    "DomainRegistry",
    "FallbackRouter",
    "JevRouter",
    "LLMRouter",
    "NullPlanner",
    "RoutingRequest",
    "Rule",
    "RuleRouter",
    "ScriptedRouter",
    "StaticPlanner",
]
