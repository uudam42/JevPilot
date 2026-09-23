"""Routing policies: ``R(S_t, C_t) → D_t``.

Reference routers (``RuleRouter``, ``ScriptedRouter``), model-backed routers
(``LLMRouter``, ``JevRouter``) that talk to injected adapters, the explicit
``FallbackRouter`` composer, and the canonical ``RoutingRequest``. Concrete
provider adapters live in :mod:`jevpilot.adapters` (offline fakes) or outside
the core package entirely; this package never imports them.
"""

from jevpilot.routing.contracts import (
    LLMAdapter,
    LLMCompletion,
    LLMPrompt,
    RoutingModelAdapter,
    RoutingModelResponse,
)
from jevpilot.routing.fallback import FallbackRouter
from jevpilot.routing.jev_router import JevRouter
from jevpilot.routing.llm_router import LLMRouter
from jevpilot.routing.model_router import ModelRouter
from jevpilot.routing.parsing import DECISION_PAYLOAD_SCHEMA, decision_from_output
from jevpilot.routing.prompts import ROUTING_PROMPT_VERSION, build_routing_prompt
from jevpilot.routing.request import (
    ActionRecord,
    CapabilityDescription,
    EvaluationSummary,
    RoutingRequest,
    StateSummary,
)
from jevpilot.routing.rules import Rule, RuleRouter
from jevpilot.routing.scripted import ScriptedRouter

__all__ = [
    "DECISION_PAYLOAD_SCHEMA",
    "ROUTING_PROMPT_VERSION",
    "ActionRecord",
    "CapabilityDescription",
    "EvaluationSummary",
    "FallbackRouter",
    "JevRouter",
    "LLMAdapter",
    "LLMCompletion",
    "LLMPrompt",
    "LLMRouter",
    "ModelRouter",
    "RoutingModelAdapter",
    "RoutingModelResponse",
    "RoutingRequest",
    "Rule",
    "RuleRouter",
    "ScriptedRouter",
    "StateSummary",
    "build_routing_prompt",
    "decision_from_output",
]
