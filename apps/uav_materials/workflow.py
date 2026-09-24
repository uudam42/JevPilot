"""One call from a natural-language request to an engineering report.

    result = run_uav_material_workflow("I need a lightweight ... panel ...")
    print(result.markdown)

The call builds a normal JevPilot run: a :class:`~jevpilot.Runtime` with the
UAV materials domain loaded, a router, and the generic controller loop. The
router chooses each next capability from the capabilities whose
preconditions hold; the domain capabilities do the work; the report is
generated from the resulting state.

Two modes, always labelled in the result and the report:

``OFFLINE_DEMO``  deterministic rule-based interpretation and a scripted reference routing
                  policy served through the real ``JevRouter`` pipeline by the offline
                  ``FakeJevAdapter``. No credentials, no network. It demonstrates the
                  orchestration; it says nothing about live-model routing quality.
``LIVE``          requirements interpreted by Claude (``AnthropicLLMAdapter``) and routing by
                  Jev (``TypeSafeJevAdapter``). Needs ``ANTHROPIC_API_KEY`` and
                  ``TYPESAFE_API_KEY``. If they are missing, the call fails with
                  :class:`LiveModeUnavailable`; it never falls back to the demo silently.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from domains.uav_materials import UAVMaterialsDomain, UAVMaterialsState
from domains.uav_materials.decision import ExistingMaterialAcceptancePolicy
from domains.uav_materials.interpretation import RequirementInterpreter, RuleBasedInterpreter
from domains.uav_materials.inverse_design import DesignSearchConfig
from domains.uav_materials.reporting import MaterialWorkflowReport, build_report, render_markdown
from domains.uav_materials.workflow import REPORT, REPORT_MARKDOWN, reference_routing_payload
from jevpilot import DefaultControlPolicy, JevRouter, Router, Runtime, WorkflowResult
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


class RunMode(StrEnum):
    LIVE = "LIVE"
    OFFLINE_DEMO = "OFFLINE_DEMO"


class LiveModeUnavailable(RuntimeError):
    """LIVE mode was requested but its credentials or SDKs are missing."""


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


# -- components ------------------------------------------------------------------------------


def offline_components() -> tuple[RequirementInterpreter, Router]:
    adapter = FakeJevAdapter(
        policy=reference_routing_payload,
        model="offline-demo-reference-policy",
        label="OFFLINE_DEMO: scripted reference routing policy (not a model)",
    )
    return RuleBasedInterpreter(), JevRouter(adapter, router_id="jev_router[offline_demo]")


def live_requirements() -> list[str]:
    """What LIVE mode still needs in this environment (empty when ready)."""
    missing = []
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        missing.append("ANTHROPIC_API_KEY (Claude requirement interpretation)")
    if not os.environ.get("TYPESAFE_API_KEY"):
        missing.append("TYPESAFE_API_KEY (Jev routing)")
    if importlib.util.find_spec("anthropic") is None:
        missing.append("the Anthropic SDK: pip install 'jevpilot[anthropic]'")
    if importlib.util.find_spec("typesafe_sdk") is None:  # what TypeSafeJevAdapter imports
        missing.append("the TypeSafe SDK: pip install 'jevpilot[jev]'")
    return missing


def live_components(*, timeout_s: float = 120.0) -> tuple[RequirementInterpreter, Router]:
    missing = live_requirements()
    if missing:
        raise LiveModeUnavailable(
            "LIVE mode needs: " + "; ".join(missing) + ". Nothing was run. Use the offline "
            "demonstration (--demo / mode=OFFLINE_DEMO) instead, or set the credentials."
        )
    from apps.uav_materials.llm_interpreter import LLMRequirementInterpreter
    from integrations.anthropic_llm import AnthropicLLMAdapter
    from integrations.typesafe_jev import TypeSafeJevAdapter

    interpreter = LLMRequirementInterpreter(AnthropicLLMAdapter(), timeout_s=timeout_s)
    router = JevRouter(TypeSafeJevAdapter(), router_id="jev_router[live]", timeout_s=timeout_s)
    return interpreter, router


# -- running ------------------------------------------------------------------------------------


def run_uav_material_workflow(
    request: str = DEMO_REQUEST,
    *,
    mode: RunMode | str = RunMode.OFFLINE_DEMO,
    policy: ExistingMaterialAcceptancePolicy | None = None,
    design_config: DesignSearchConfig | None = None,
    max_steps: int = 30,
) -> UAVWorkflowResult:
    """Natural-language request → engineering report (see the module docstring)."""
    mode = RunMode(mode)
    if mode is RunMode.LIVE:
        interpreter, router = live_components()
        note = "requirements interpreted by a language model and routing by Jev (live services)"
    else:
        interpreter, router = offline_components()
        note = OFFLINE_NOTE
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
