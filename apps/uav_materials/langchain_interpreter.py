"""Requirement interpretation through LangChain (``--llm-backend langchain``).

:class:`LangChainRequirementInterpreter` implements the same
:class:`~domains.uav_materials.interpretation.RequirementInterpreter` contract
as the native :class:`~apps.uav_materials.llm_interpreter.LLMRequirementInterpreter`
and reuses everything that matters for correctness: the domain's prompt, the
same JSON schema (``InterpretationPayload``: no field for any material
property) requested through LangChain's structured output (tool calling), and
the same validation (:func:`~domains.uav_materials.interpretation.parse_interpretation`:
verbatim quotes, numbers and units written by the user).

LangChain only carries the request to a chat model and the structured answer
back. It makes no routing decision; Jev (or another JevPilot router) routes.
If the call fails or the answer has no structured output, the interpretation
fails visibly; nothing falls back to another interpreter.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from domains.uav_materials.interpretation import (
    INTERPRETATION_PROMPT_VERSION,
    InterpretationOutcome,
    InterpretationPayload,
    RuleBasedInterpreter,
    interpretation_system_prompt,
    interpretation_user_prompt,
    parse_interpretation,
)
from integrations.langchain_chat import ScriptedToolChatModel, describe_chat_model
from integrations.redaction import redact


class LangChainInterpreterError(RuntimeError):
    """The LangChain chat-model call failed (message redacted)."""


class LangChainRequirementInterpreter:
    name = "langchain_interpreter"

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        self.schema = InterpretationPayload.model_json_schema()  # the native path's schema
        self._structured = chat_model.with_structured_output(self.schema, include_raw=True)

    def describe(self) -> dict[str, Any]:
        scripted = isinstance(self.chat_model, ScriptedToolChatModel)
        return {
            "kind": "scripted" if scripted else "llm",
            "name": self.name,
            "backend": "langchain",
            "prompt_version": INTERPRETATION_PROMPT_VERSION,
            "structured_output": "with_structured_output (tool calling)",
            "chat_model": describe_chat_model(self.chat_model),
            "langchain_core": importlib.metadata.version("langchain-core"),
        }

    def messages(self, request_text: str) -> list[BaseMessage]:
        return [
            SystemMessage(interpretation_system_prompt()),
            HumanMessage(interpretation_user_prompt(request_text)),
        ]

    def interpret(self, request_text: str) -> InterpretationOutcome:
        try:
            result = self._structured.invoke(self.messages(request_text))
        except Exception as exc:  # provider, network, auth: the capability records it
            raise LangChainInterpreterError(
                f"LangChain chat model call failed: {type(exc).__name__}: {redact(exc)[:500]}"
            ) from exc
        out: dict[str, Any] = result if isinstance(result, dict) else {"parsed": result}
        raw, parsed = out.get("raw"), out.get("parsed")
        meta = getattr(raw, "response_metadata", None) or {}
        described = {**self.describe(), "served_model": meta.get("model_name") or meta.get("model")}
        if isinstance(parsed, dict):
            return parse_interpretation(parsed, request_text, described)
        problem = out.get("parsing_error") or "the model returned no structured output"
        return InterpretationOutcome(
            request_text=request_text,
            interpreter=described,
            status="failed",
            error=f"structured output missing or invalid: {redact(problem)[:500]}",
            raw_output=redact(getattr(raw, "content", ""))[:4000],
        )


def request_text_of(messages: list[BaseMessage]) -> str:
    """The user's request inside the interpretation prompt (inverse of the user prompt)."""
    user = next(str(m.content) for m in messages if isinstance(m, HumanMessage))
    probe = "\x00"
    head, _, tail = interpretation_user_prompt(probe).partition(probe)
    if user.startswith(head) and user.endswith(tail):
        return user[len(head) : len(user) - len(tail)]
    raise ValueError("not an interpretation prompt")


def offline_langchain_interpreter() -> LangChainRequirementInterpreter:
    """LangChain path with a deterministic scripted chat model (OFFLINE_DEMO, not a language model).

    The scripted model answers the structured-output request with the rule-based
    phrase parser's payload, so the LangChain plumbing (messages, tool-calling
    structured output, parsing, validation) runs end to end without a network.
    """
    rules = RuleBasedInterpreter()
    model = ScriptedToolChatModel(
        policy=lambda messages: rules.payload(request_text_of(messages)),
        model_name="scripted-rule-based-phrases",
        label="OFFLINE: scripted chat model serving the rule-based phrase parser "
        "(not a language model)",
    )
    return LangChainRequirementInterpreter(model)
