"""LangChain chat models for JevPilot (optional; ``pip install 'jevpilot[langchain]'``).

LangChain is used only as a model-communication and structured-output layer.
It does not route, orchestrate or execute anything: routing stays with Jev
(or another JevPilot router) and the controller loop stays in ``jevpilot``.
Only ``langchain-core`` and a provider package are used (no ``langchain``
meta-package, no LangGraph). Neither the core nor any domain imports this module.

Two chat models:

``chat_model(spec)``         a real provider model, e.g. ``"anthropic:claude-opus-5"``.
                             The spec comes from the argument, else ``JEVPILOT_LANGCHAIN_MODEL``,
                             else Anthropic with the native adapter's model setting.
``ScriptedToolChatModel``    an offline, deterministic stand-in that answers structured-output
                             (tool-calling) requests from a Python policy. Not a language model;
                             for demos and tests only, and always labelled as such.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

from integrations.anthropic_llm import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from integrations.anthropic_llm import MODEL_ENV as ANTHROPIC_MODEL_ENV

SPEC_ENV = "JEVPILOT_LANGCHAIN_MODEL"


@dataclass(frozen=True)
class Provider:
    package: str  # the LangChain provider package (import name)
    chat_class: str
    key_envs: tuple[str, ...]  # any one of them must be set
    default_model: Callable[[], str]
    install: str


# Adding a provider is one entry here; the interpreter code does not change.
PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        package="langchain_anthropic",
        chat_class="ChatAnthropic",
        key_envs=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        default_model=lambda: (
            os.environ.get(ANTHROPIC_MODEL_ENV, "").strip() or ANTHROPIC_DEFAULT_MODEL
        ),
        install="pip install 'jevpilot[langchain]'",
    ),
}


def parse_spec(spec: str | None = None) -> tuple[str, str]:
    """``"provider:model"`` (or ``"provider"``) → (provider, model)."""
    text = (spec or os.environ.get(SPEC_ENV, "") or "anthropic").strip()
    provider, _, model = text.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown LangChain provider {provider!r}; supported: {', '.join(sorted(PROVIDERS))}"
        )
    return provider, model.strip() or PROVIDERS[provider].default_model()


def missing_requirements(spec: str | None = None) -> list[str]:
    """What a live LangChain chat model still needs here (empty when ready). Never shows values."""
    provider, _ = parse_spec(spec)
    p = PROVIDERS[provider]
    missing = []
    if importlib.util.find_spec(p.package) is None:
        missing.append(f"the LangChain {provider} package: {p.install}")
    if not any(os.environ.get(k, "").strip() for k in p.key_envs):
        missing.append(f"{p.key_envs[0]} (LangChain {provider} chat model)")
    return missing


def chat_model(
    spec: str | None = None, *, timeout_s: float = 120.0, max_retries: int = 2
) -> BaseChatModel:
    """A real provider chat model. Credentials are read by the provider SDK from the environment."""
    provider, model = parse_spec(spec)
    p = PROVIDERS[provider]
    cls: Any = getattr(importlib.import_module(p.package), p.chat_class)
    built: BaseChatModel = cls(
        model=model, max_tokens=8000, default_request_timeout=timeout_s, max_retries=max_retries
    )
    return built


def describe_chat_model(model: BaseChatModel) -> dict[str, Any]:
    """Identity of a chat model for provenance (class, provider type, model name; no secrets)."""
    name = getattr(model, "model", None) or getattr(model, "model_name", None)
    return {
        "class": type(model).__name__,
        "llm_type": model._llm_type,
        "model": str(name) if name else None,
        "label": getattr(model, "label", None),
    }


class ScriptedToolChatModel(BaseChatModel):
    """Offline, deterministic chat model: answers a tool call with ``policy(messages)``.

    Supports ``with_structured_output`` (tool calling). Without bound tools it
    answers with the policy's result as JSON text. It never calls a network.
    """

    policy: Callable[[list[BaseMessage]], dict[str, Any]]
    model_name: str = "scripted-offline"
    label: str = "OFFLINE: deterministic scripted chat model (not a language model)"

    @property
    def _llm_type(self) -> str:
        return "jevpilot-scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        args = self.policy(messages)
        tools = kwargs.get("tools") or []
        if tools:
            name = tools[0]["function"]["name"]
            message = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": "scripted_0", "type": "tool_call"}],
                response_metadata={"model_name": self.model_name},
            )
        else:
            message = AIMessage(
                content=json.dumps(args), response_metadata={"model_name": self.model_name}
            )
        return ChatResult(generations=[ChatGeneration(message=message)])
