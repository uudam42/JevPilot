"""One call from a natural-language request to an engineering report.

    result = run_uav_material_workflow("I need a lightweight ... panel ...")
    print(result.markdown)

The call builds a normal JevPilot run: a :class:`~jevpilot.Runtime` with the
UAV materials domain loaded, a router, and the generic controller loop. The
router chooses each next capability from the capabilities whose
preconditions hold; the domain capabilities do the work; the report is
generated from the resulting state.

Three modes, always labelled in the result and the report:

``OFFLINE_DEMO``  deterministic offline interpretation and a scripted reference routing
                  policy served through the real ``JevRouter`` pipeline by the offline
                  ``FakeJevAdapter``. No credentials, no network. It demonstrates the
                  orchestration; it says nothing about live-model routing quality.
``LIVE``          requirements interpreted by a language model and routing by Jev
                  (``TypeSafeJevAdapter``). Needs ``ANTHROPIC_API_KEY`` and
                  ``TYPESAFE_API_KEY``.
``LIVE_ROUTING``  routing by the real Jev service; requirements interpreted offline (the
                  deterministic parser, or a scripted chat model through LangChain). Needs
                  only ``TYPESAFE_API_KEY``. Not fully live, and labelled so.

The LLM backend (``native`` or ``langchain``) selects how requirements reach a chat
model: the native Anthropic adapter, or LangChain structured output. Both use the same
prompt, schema and validation. LangChain never routes.

If a live mode's credentials, SDKs or Jev preflight are missing, the call fails with
:class:`LiveModeUnavailable` before anything runs; it never falls back to the demo.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from domains.uav_materials import UAVMaterialsDomain, UAVMaterialsState
from domains.uav_materials.decision import ExistingMaterialAcceptancePolicy
from domains.uav_materials.interpretation import RequirementInterpreter, RuleBasedInterpreter
from domains.uav_materials.inverse_design import DesignSearchConfig
from domains.uav_materials.reporting import MaterialWorkflowReport, build_report, render_markdown
from domains.uav_materials.workflow import REPORT, REPORT_MARKDOWN, reference_routing_payload
from jevpilot import (
    DefaultControlPolicy,
    JevRouter,
    Router,
    Runtime,
    TraceEvent,
    TraceEventType,
    WorkflowResult,
)
from jevpilot.adapters import FakeJevAdapter

DEMO_REQUEST = (
    "I need a lightweight structural material for a small UAV wing-skin panel with low "
    "density, high in-plane stiffness in the x and y directions, and good load-carrying "
    "capability. The density must not exceed 1800 kg/m^3. The panel needs an in-plane "
    "stiffness of at least 40 GPa in both the x and y directions and an in-plane shear "
    "stiffness of at least 15 GPa. It will operate near the sea. Find the closest existing "
    "material. If no existing material is sufficiently suitable, design a composite candidate "
    "and explain the result."
)
OFFLINE_NOTE = (
    "deterministic rule-based interpretation and a scripted reference routing policy served "
    "through the JevRouter pipeline by the offline FakeJevAdapter; no language model or Jev "
    "service was called, and this run says nothing about live-model routing quality"
)
OFFLINE_LANGCHAIN_NOTE = (
    "deterministic scripted chat model (serving the rule-based phrase parser) reached through "
    "LangChain structured output, and a scripted reference routing policy served through the "
    "JevRouter pipeline by the offline FakeJevAdapter; no language model or Jev service was "
    "called, and this run says nothing about live-model routing quality"
)


class RunMode(StrEnum):
    LIVE = "LIVE"
    LIVE_ROUTING = "LIVE_ROUTING"
    OFFLINE_DEMO = "OFFLINE_DEMO"


class LLMBackend(StrEnum):
    NATIVE = "native"
    LANGCHAIN = "langchain"


class ComponentUnavailable(RuntimeError):
    """A requested component (backend, SDK, credential) is not available here."""


class LiveModeUnavailable(ComponentUnavailable):
    """A live mode was requested but its credentials, SDKs or Jev preflight are missing."""


@dataclass(frozen=True)
class UAVWorkflowResult:
    mode: RunMode
    report: MaterialWorkflowReport
    markdown: str
    state: UAVMaterialsState
    run: WorkflowResult

    @property
    def decision(self) -> str | None:
        return self.report.decision.outcome if self.report.decision else None

    @property
    def complete(self) -> bool:
        return self.report.status == "complete" and self.run.succeeded

    def to_json(self, indent: int | None = 2) -> str:
        return self.report.model_dump_json(indent=indent)

    def routing_log(self) -> list[dict[str, Any]]:
        return routing_log(self.run.trace)


def routing_log(trace: Sequence[TraceEvent]) -> list[dict[str, Any]]:
    """One sanitized record per routing decision or failure.

    Only these fields are kept: timestamp, step, router, model, intent, capability,
    confidence, latency_ms, status and retries. No state, request, prompt, header or
    credential is ever included.
    """
    records = []
    for event in trace:
        if event.type not in (TraceEventType.ROUTING_DECISION, TraceEventType.ROUTING_FAILURE):
            continue
        payload = event.payload
        attempts = payload.get("attempts") or []
        last = attempts[-1] if attempts else {}
        meta = last.get("metadata") or {}
        decision = payload.get("decision") or {}
        error = payload.get("error") or {}
        status = (
            f"error:{error.get('type')}"
            if error
            else "fallback"
            if payload.get("fallback_used")
            else "ok"
        )
        records.append(
            {
                "timestamp": event.timestamp.isoformat(),
                "step": event.step,
                "router": decision.get("router_id") or last.get("router_id"),
                "model": last.get("model") or meta.get("requested_model"),
                "intent": decision.get("intent"),
                "capability": decision.get("capability_id"),
                "confidence": decision.get("confidence"),
                "latency_ms": round(float(payload.get("routing_latency_s") or 0.0) * 1000, 1),
                "status": status,
                "retries": sum(
                    int((a.get("metadata") or {}).get("retries") or 0) for a in attempts
                ),
            }
        )
    return records


# -- components ------------------------------------------------------------------------------


def offline_interpreter(
    llm_backend: LLMBackend | str = LLMBackend.NATIVE,
) -> RequirementInterpreter:
    """The deterministic interpreter for a backend (no language model, no network)."""
    if LLMBackend(llm_backend) is LLMBackend.NATIVE:
        return RuleBasedInterpreter()
    if importlib.util.find_spec("langchain_core") is None:
        raise ComponentUnavailable(
            "the LangChain backend needs langchain-core: pip install 'jevpilot[langchain]'"
        )
    from apps.uav_materials.langchain_interpreter import offline_langchain_interpreter

    return offline_langchain_interpreter()


def offline_components(
    llm_backend: LLMBackend | str = LLMBackend.NATIVE,
) -> tuple[RequirementInterpreter, Router]:
    adapter = FakeJevAdapter(
        policy=reference_routing_payload,
        model="offline-demo-reference-policy",
        label="OFFLINE_DEMO: scripted reference routing policy (not a model)",
    )
    router = JevRouter(adapter, router_id="jev_router[offline_demo]")
    return offline_interpreter(llm_backend), router


def live_requirements(
    llm_backend: LLMBackend | str = LLMBackend.NATIVE, *, routing_only: bool = False
) -> list[str]:
    """What a live mode still needs in this environment (empty when ready). Never shows values."""
    missing = []
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        missing.append("TYPESAFE_API_KEY (Jev routing)")
    if importlib.util.find_spec("typesafe_sdk") is None:  # what TypeSafeJevAdapter imports
        missing.append("the TypeSafe SDK: pip install 'jevpilot[jev]'")
    backend = LLMBackend(llm_backend)
    if backend is LLMBackend.LANGCHAIN and importlib.util.find_spec("langchain_core") is None:
        missing.append("langchain-core: pip install 'jevpilot[langchain]'")
    if routing_only:
        return missing
    if backend is LLMBackend.LANGCHAIN:
        if importlib.util.find_spec("langchain_core") is not None:
            from integrations.langchain_chat import missing_requirements

            missing += missing_requirements()
    else:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            missing.append("ANTHROPIC_API_KEY (requirement interpretation)")
        if importlib.util.find_spec("anthropic") is None:
            missing.append("the Anthropic SDK: pip install 'jevpilot[anthropic]'")
    return missing


def live_router(*, timeout_s: float = 120.0) -> tuple[Router, dict[str, Any]]:
    """JevRouter over the real Jev service, after a preflight (authentication, model)."""
    from integrations.typesafe_jev import TypeSafeJevAdapter
    from jevpilot.exceptions import RoutingError

    adapter = TypeSafeJevAdapter()
    try:
        preflight = adapter.verify()
    except RoutingError as exc:
        raise LiveModeUnavailable(f"Jev preflight failed: {exc}. Nothing was run.") from exc
    return JevRouter(adapter, router_id="jev_router[live]", timeout_s=timeout_s), preflight


def live_components(
    *,
    llm_backend: LLMBackend | str = LLMBackend.NATIVE,
    routing_only: bool = False,
    timeout_s: float = 120.0,
) -> tuple[RequirementInterpreter, Router]:
    backend = LLMBackend(llm_backend)
    missing = live_requirements(backend, routing_only=routing_only)
    if missing:
        mode = RunMode.LIVE_ROUTING if routing_only else RunMode.LIVE
        raise LiveModeUnavailable(
            f"{mode} mode needs: " + "; ".join(missing) + ". Nothing was run. Use the offline "
            "demonstration (--demo / mode=OFFLINE_DEMO) instead, or set the credentials."
        )
    interpreter: RequirementInterpreter
    if routing_only:
        interpreter = offline_interpreter(backend)
    elif backend is LLMBackend.LANGCHAIN:
        from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter
        from integrations.langchain_chat import chat_model

        interpreter = LangChainRequirementInterpreter(chat_model(timeout_s=timeout_s))
    else:
        from apps.uav_materials.llm_interpreter import LLMRequirementInterpreter
        from integrations.anthropic_llm import AnthropicLLMAdapter

        interpreter = LLMRequirementInterpreter(AnthropicLLMAdapter(), timeout_s=timeout_s)
    router, _ = live_router(timeout_s=timeout_s)
    return interpreter, router


# -- running ------------------------------------------------------------------------------------


def run_uav_material_workflow(
    request: str = DEMO_REQUEST,
    *,
    mode: RunMode | str = RunMode.OFFLINE_DEMO,
    llm_backend: LLMBackend | str = LLMBackend.NATIVE,
    policy: ExistingMaterialAcceptancePolicy | None = None,
    design_config: DesignSearchConfig | None = None,
    max_steps: int = 30,
) -> UAVWorkflowResult:
    """Natural-language request → engineering report (see the module docstring)."""
    mode, backend = RunMode(mode), LLMBackend(llm_backend)
    via = "through LangChain structured output" if backend is LLMBackend.LANGCHAIN else "natively"
    if mode is RunMode.LIVE:
        interpreter, router = live_components(llm_backend=backend)
        note = (
            f"requirements interpreted by a language model ({via}) and routing by Jev "
            "(live services)"
        )
    elif mode is RunMode.LIVE_ROUTING:
        interpreter, router = live_components(llm_backend=backend, routing_only=True)
        note = (
            "routing by the real Jev service; requirements interpreted offline by "
            + (
                "a deterministic scripted chat model through LangChain"
                if backend is LLMBackend.LANGCHAIN
                else "the deterministic phrase parser"
            )
            + " (no language model was called); this run is not fully live"
        )
    else:
        interpreter, router = offline_components(backend)
        note = OFFLINE_NOTE if backend is LLMBackend.NATIVE else OFFLINE_LANGCHAIN_NOTE
    return execute_workflow(
        request,
        interpreter=interpreter,
        router=router,
        mode=mode,
        note=note,
        policy=policy,
        design_config=design_config,
        max_steps=max_steps,
    )


def execute_workflow(
    request: str,
    *,
    interpreter: RequirementInterpreter,
    router: Router,
    mode: RunMode,
    note: str = "",
    policy: ExistingMaterialAcceptancePolicy | None = None,
    design_config: DesignSearchConfig | None = None,
    max_steps: int = 30,
) -> UAVWorkflowResult:
    """Run the workflow with explicit components (the building block of both modes)."""
    runtime = Runtime()
    domain = UAVMaterialsDomain(interpreter=interpreter, policy=policy, design_config=design_config)
    runtime.load(domain)
    controller = runtime.controller(
        router, domains=[domain.name], policy=DefaultControlPolicy(max_steps=max_steps)
    )
    run_info: dict[str, Any] = {
        "mode": mode.value,
        "note": note,
        "router": router.config(),
        "interpreter": interpreter.describe(),
    }
    result = controller.run(domain.new_request_workflow(request, run_info=run_info))
    state = result.state
    assert isinstance(state, UAVMaterialsState)
    report_artifact = state.latest_artifact(REPORT)
    markdown_artifact = state.latest_artifact(REPORT_MARKDOWN)
    if report_artifact is not None and markdown_artifact is not None:
        report: MaterialWorkflowReport = report_artifact.content
        markdown = str(markdown_artifact.content)
    else:  # the loop stopped early: report what was computed, and say why
        reason = (
            f"the workflow ended with status '{state.status}' before the report step: "
            f"{result.control.reason}"
        )
        report = build_report(state, incomplete_reason=reason)
        markdown = render_markdown(report)
    return UAVWorkflowResult(mode=mode, report=report, markdown=markdown, state=state, run=result)
