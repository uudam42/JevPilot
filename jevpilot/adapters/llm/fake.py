"""Offline stand-in for a text-completion LLM. No network, no model, fully deterministic."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

from jevpilot.adapters.scripting import Policy, Script, ScriptItem
from jevpilot.core import ModelUsage
from jevpilot.routing.contracts import LLMCompletion, LLMPrompt, RoutingModelResponse
from jevpilot.routing.request import RoutingRequest


class FakeLLMAdapter:
    """Implements :class:`~jevpilot.routing.contracts.LLMAdapter` from a script or policy.

    Like a real LLM it only sees the prompt text. Policies receive the
    :class:`RoutingRequest` recovered from the JSON embedded in the prompt,
    so the same request-level policy can drive both fake adapters. Dict items
    are returned as JSON text, and string items verbatim.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        policy: Policy | None = None,
        on_exhausted: str = "error",
        model: str = "fake-llm",
        usage: ModelUsage | None = None,
        label: str | None = None,
    ) -> None:
        self._script = Script(script, policy=policy, on_exhausted=on_exhausted)
        self.model = model
        self.usage = usage
        self.label = label
        self.prompts: list[LLMPrompt] = []

    def complete(self, prompt: LLMPrompt, *, timeout_s: float | None = None) -> LLMCompletion:
        self.prompts.append(prompt)
        item = self._script.next(request_from_prompt(prompt))
        if isinstance(item, BaseException):
            raise copy.copy(item)  # fresh instance: scripts may reuse one exception
        if isinstance(item, LLMCompletion):
            return item
        if isinstance(item, RoutingModelResponse):
            item = item.output
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMCompletion(text=text, model=self.model, usage=self.usage, stop_reason="end_turn")

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "kind": "llm",
            "model": self.model,
            "label": self.label,
            **self._script.describe(),
        }


def request_from_prompt(prompt: LLMPrompt) -> RoutingRequest:
    """Recover the routing request embedded in a prompt built by ``build_routing_prompt``."""
    start = prompt.user.find("{")
    if start < 0:
        raise ValueError("prompt does not embed a JSON routing request")
    return RoutingRequest.model_validate(json.loads(prompt.user[start:]))
