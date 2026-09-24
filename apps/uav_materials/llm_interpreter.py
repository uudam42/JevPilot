"""Requirement interpretation with a real language model (LIVE mode).

The model receives the domain's interpretation prompt and returns JSON. The
answer is then validated by exactly the same rules as the offline parser
(:func:`~domains.uav_materials.interpretation.parse_interpretation`): items
must quote the request, numbers and units must be written by the user, and
the schema has no field for any material property. A failed call or an
invalid answer becomes a ``failed`` interpretation that the report explains.
"""

from __future__ import annotations

from typing import Any

from domains.uav_materials.interpretation import (
    INTERPRETATION_PROMPT_VERSION,
    InterpretationOutcome,
    InterpretationPayload,
    interpretation_system_prompt,
    interpretation_user_prompt,
    parse_interpretation,
)
from jevpilot.routing import LLMAdapter, LLMPrompt


class LLMRequirementInterpreter:
    name = "llm_interpreter"

    def __init__(self, adapter: LLMAdapter, *, timeout_s: float | None = 120.0) -> None:
        self.adapter = adapter
        self.timeout_s = timeout_s

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "llm",
            "name": self.name,
            "prompt_version": INTERPRETATION_PROMPT_VERSION,
            "adapter": self.adapter.describe(),
        }

    def prompt(self, request_text: str) -> LLMPrompt:
        return LLMPrompt(
            system=interpretation_system_prompt(),
            user=interpretation_user_prompt(request_text),
            response_schema=InterpretationPayload.model_json_schema(),
            version=INTERPRETATION_PROMPT_VERSION,
        )

    def interpret(self, request_text: str) -> InterpretationOutcome:
        completion = self.adapter.complete(self.prompt(request_text), timeout_s=self.timeout_s)
        described = {**self.describe(), "served_model": completion.model}
        return parse_interpretation(_json_object(completion.text), request_text, described)


def _json_object(text: str) -> str:
    """The outermost JSON object in the answer (tolerates a code fence around it)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text
