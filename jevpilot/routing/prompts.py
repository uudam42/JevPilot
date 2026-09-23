"""Prompt construction for :class:`~jevpilot.routing.llm_router.LLMRouter`.

All LLM routing prompt text lives here, versioned by :data:`ROUTING_PROMPT_VERSION`
so benchmark runs can record exactly which prompt produced a decision. The
prompt contains no domain knowledge: everything specific to a workflow
arrives through the :class:`RoutingRequest`. Jev does not use this module;
its input formatting belongs to its adapter.
"""

from __future__ import annotations

import json

from jevpilot.routing.contracts import LLMPrompt
from jevpilot.routing.parsing import DECISION_PAYLOAD_SCHEMA
from jevpilot.routing.request import RoutingRequest

ROUTING_PROMPT_VERSION = "routing-prompt/1"

SYSTEM_PROMPT = """\
You are the routing policy of a workflow orchestration system. Given the current \
workflow state and the capabilities available right now, choose the single next \
action. You only decide. You do not execute anything, and nothing you write is run \
as code.

Rules:
1. Choose only from the capabilities listed in available_capabilities, using their \
exact id. Never invent a capability.
2. Provide inputs that satisfy the chosen capability's input_schema exactly: every \
required field present, correct types, no extra fields unless the schema allows them.
3. Base the decision on the current state: goal, context, previous_actions (including \
failures and their errors), previous_evaluations and the plan, if any.
4. Avoid unnecessary actions. Do not repeat a step that already succeeded unless the \
state shows it must be redone. Prefer the shortest path to the goal.
5. If the goal already appears to be met, use action "finish".
6. If progress needs information that only a human can provide and no available \
capability can obtain it, use action "ask_human".
7. Report confidence honestly in [0, 1], or null if you cannot judge.

Respond with exactly one JSON object and nothing else (no prose, no code fences), \
matching this JSON Schema:
{schema}"""


def build_routing_prompt(request: RoutingRequest) -> LLMPrompt:
    """Render a routing request as a provider-neutral prompt."""
    schema = json.dumps(DECISION_PAYLOAD_SCHEMA, sort_keys=True)
    return LLMPrompt(
        system=SYSTEM_PROMPT.format(schema=schema),
        user="Routing request (JSON):\n" + request.to_json(indent=2),
        response_schema=DECISION_PAYLOAD_SCHEMA,
        version=ROUTING_PROMPT_VERSION,
    )
