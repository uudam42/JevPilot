"""Turning untrusted model output into a :class:`RoutingDecision`.

Model output is data, never instructions to the framework. The parser accepts
exactly one JSON object with a fixed set of keys (:data:`DECISION_PAYLOAD_SCHEMA`),
rejects anything else as :class:`MalformedRoutingDecisionError`, and never
evaluates, imports or instantiates anything. Whether the chosen capability
exists and its inputs fit the schema is checked afterwards by
:func:`~jevpilot.interfaces.router.validate_decision`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from jevpilot.core import RoutingDecision, RoutingIntent
from jevpilot.exceptions import MalformedRoutingDecisionError
from jevpilot.interfaces.router import json_data_problem

ALLOWED_KEYS = frozenset({"action", "capability_id", "inputs", "reason", "confidence"})
ACTIONS = {
    "invoke": RoutingIntent.INVOKE,
    "finish": RoutingIntent.FINISH,
    "ask_human": RoutingIntent.ASK_HUMAN,
}
MAX_REASON_CHARS = 2000

DECISION_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": sorted(ACTIONS),
            "description": "invoke a capability, finish (goal met), or ask_human "
            "(information only a human can provide is missing)",
        },
        "capability_id": {
            "type": ["string", "null"],
            "description": "id of one available capability when action is invoke, else null",
        },
        "inputs": {
            "type": "object",
            "description": "inputs matching the capability's input_schema; {} otherwise",
        },
        "reason": {"type": "string", "description": "short justification"},
        "confidence": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "description": "self-reported confidence in [0, 1], or null",
        },
    },
    "required": ["action", "capability_id", "inputs", "reason"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


def decode_output(output: Any) -> dict[str, Any]:
    """Raw model output (text or mapping) → a JSON object, or a malformed-decision error."""
    if isinstance(output, str):
        text = output.strip()
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            output = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _malformed(f"response is not valid JSON ({exc.msg})", output) from None
    if not isinstance(output, Mapping):
        raise _malformed(f"expected a JSON object, got {type(output).__name__}", output)
    problem = json_data_problem(output, "response")
    if problem is not None:
        raise _malformed(problem, None)
    return dict(output)


def decision_from_output(output: Any, *, router_id: str) -> RoutingDecision:
    """Parse and structurally check a raw decision payload."""
    payload = decode_output(output)
    unknown = sorted(set(payload) - ALLOWED_KEYS)
    if unknown:
        raise _malformed(f"unexpected keys {unknown}", payload)

    capability_id = payload.get("capability_id")
    action = payload.get("action", "invoke" if capability_id is not None else None)
    if action not in ACTIONS:
        raise _malformed(f"action must be one of {sorted(ACTIONS)}, got {action!r}", payload)
    intent = ACTIONS[action]

    inputs = payload.get("inputs", {})
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        raise _malformed("inputs must be a JSON object", payload)

    if intent is RoutingIntent.INVOKE:
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise _malformed("action 'invoke' requires a non-empty string capability_id", payload)
    elif capability_id is not None or inputs:
        raise _malformed(f"action {action!r} must have capability_id null and no inputs", payload)

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise _malformed("reason must be a string", payload)

    confidence = payload.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise _malformed("confidence must be a number or null", payload)
        if not 0.0 <= float(confidence) <= 1.0:
            raise _malformed("confidence must lie in [0, 1]", payload)
        confidence = float(confidence)

    return RoutingDecision(
        capability_id=capability_id,
        intent=intent,
        inputs=inputs,
        reason=reason[:MAX_REASON_CHARS],
        confidence=confidence,
        router_id=router_id,
    )


def _malformed(message: str, output: Any) -> MalformedRoutingDecisionError:
    details: dict[str, Any] = {}
    if output is not None:
        excerpt = output if isinstance(output, str) else json.dumps(output, default=repr)
        details["response_excerpt"] = excerpt[:300]
    return MalformedRoutingDecisionError(f"malformed routing decision: {message}", details=details)
