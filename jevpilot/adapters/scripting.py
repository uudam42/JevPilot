"""Scripted model behaviour for offline fake adapters.

A *script item* describes one model response:

- ``dict``: a decision payload (serialised to JSON text for LLM fakes);
- ``str``: raw model text, returned verbatim (use it for malformed output);
- an exception instance: raised by the adapter (timeouts, provider errors);
- ``RoutingModelResponse`` / ``LLMCompletion``: returned as-is (e.g. to report usage);
- a callable ``(RoutingRequest) -> item``: computed per request.

The helpers below build the standard reliability cases. :func:`with_faults`
wraps a policy with seeded, reproducible fault injection.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

from jevpilot.exceptions import RouterTimeoutError
from jevpilot.routing.contracts import LLMCompletion, RoutingModelResponse
from jevpilot.routing.request import RoutingRequest

ScriptItem: TypeAlias = (
    dict[str, Any]
    | str
    | BaseException
    | RoutingModelResponse
    | LLMCompletion
    | Callable[[RoutingRequest], Any]
)
Policy: TypeAlias = Callable[[RoutingRequest], Any]


class ScriptExhaustedError(RuntimeError):
    """A scripted fake was asked for more responses than it was given."""


def valid(
    capability_id: str,
    inputs: Mapping[str, Any] | None = None,
    *,
    reason: str = "scripted decision",
    confidence: float | None = None,
) -> dict[str, Any]:
    return {
        "action": "invoke",
        "capability_id": capability_id,
        "inputs": dict(inputs or {}),
        "reason": reason,
        "confidence": confidence,
    }


def finish(reason: str = "goal met", confidence: float | None = None) -> dict[str, Any]:
    return {
        "action": "finish",
        "capability_id": None,
        "inputs": {},
        "reason": reason,
        "confidence": confidence,
    }


def ask_human(
    reason: str = "missing information", confidence: float | None = None
) -> dict[str, Any]:
    return {
        "action": "ask_human",
        "capability_id": None,
        "inputs": {},
        "reason": reason,
        "confidence": confidence,
    }


def invalid_capability(capability_id: str = "nonexistent.capability") -> dict[str, Any]:
    return valid(capability_id, reason="scripted: capability that is not offered")


def invalid_inputs(capability_id: str, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A real capability with inputs that should fail its schema (caller may supply them)."""
    bad = dict(inputs) if inputs is not None else {"__unexpected__": object.__name__}
    return valid(capability_id, bad, reason="scripted: schema-violating inputs")


def low_confidence(
    capability_id: str, inputs: Mapping[str, Any] | None = None, confidence: float = 0.05
) -> dict[str, Any]:
    return valid(capability_id, inputs, reason="scripted: low confidence", confidence=confidence)


def malformed(text: str = "I think you should probably run the parser next.") -> str:
    return text


def timeout(message: str = "simulated model timeout") -> RouterTimeoutError:
    return RouterTimeoutError(message)


def adapter_error(message: str = "simulated provider outage") -> RuntimeError:
    return RuntimeError(message)


FAULTS: dict[str, Callable[[RoutingRequest], Any]] = {
    "malformed": lambda r: malformed(),
    "invalid_capability": lambda r: invalid_capability(),
    "invalid_inputs": lambda r: invalid_inputs(r.capability_ids[0]) if r.capability_ids else {},
    "timeout": lambda r: timeout(),
    "adapter_error": lambda r: adapter_error(),
}


def with_faults(policy: Policy, rates: Mapping[str, float], seed: int = 0) -> Policy:
    """Wrap ``policy``: each call's answer becomes fault ``k`` with probability ``rates[k]``.

    The random stream is seeded, so a sequence of requests always meets the
    same faults: benchmark runs with fake adapters are reproducible.
    """
    unknown = set(rates) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown fault kinds {sorted(unknown)}; known: {sorted(FAULTS)}")
    if sum(rates.values()) > 1.0:
        raise ValueError("fault rates must sum to at most 1")
    rng = random.Random(seed)
    ordered = sorted(rates.items())

    def faulty(request: RoutingRequest) -> Any:
        draw, acc = rng.random(), 0.0
        for kind, rate in ordered:
            acc += rate
            if draw < acc:
                return FAULTS[kind](request)
        return policy(request)

    return faulty


class Script:
    """Cycles through a fixed sequence of items, or computes them with a policy."""

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        policy: Policy | None = None,
        on_exhausted: str = "error",
    ) -> None:
        if (script is None) == (policy is None):
            raise ValueError("give exactly one of script= or policy=")
        if on_exhausted not in {"error", "finish", "repeat_last"}:
            raise ValueError("on_exhausted must be 'error', 'finish' or 'repeat_last'")
        self.items = list(script or ())
        self.policy = policy
        self.on_exhausted = on_exhausted
        self.calls = 0

    def next(self, request: RoutingRequest) -> Any:
        """The resolved item for this call (callables applied, not yet raised)."""
        index = self.calls
        self.calls += 1
        if self.policy is not None:
            item: Any = self.policy(request)
        elif index < len(self.items):
            item = self.items[index]
        elif self.on_exhausted == "finish":
            item = finish("script exhausted")
        elif self.on_exhausted == "repeat_last" and self.items:
            item = self.items[-1]
        else:
            raise ScriptExhaustedError(f"script of {len(self.items)} items exhausted")
        while callable(item) and not isinstance(item, BaseException):
            item = item(request)
        return item

    def describe(self) -> dict[str, Any]:
        mode = "policy" if self.policy is not None else "script"
        return {"mode": mode, "script_length": len(self.items), "on_exhausted": self.on_exhausted}
