"""Benchmark fixture schema and loaders.

Two fixture kinds share one file format (see ``BENCHMARKING.md``):

* **Decision cases** evaluate a single routing step ``R(S_t, C_t) → D_t``.
  Each case holds a compact state, the capabilities offered at that step
  (inline JSON, or ids from the file's ``capability_catalog``) and the
  expected behaviour. Several actions may be acceptable.
* **Workflow cases** run a whole workflow of an executable suite
  (``experiments.routing.suites``) and check its terminal status.

Fixtures are plain JSON, so they can be written by hand, generated, or
replayed against any router, including future learned ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from jevpilot import (
    Artifact,
    CapabilitySpec,
    ControlAction,
    ControlDecision,
    ErrorInfo,
    EvaluationResult,
    Goal,
    HistoryEntry,
    Observation,
    Provenance,
    RoutingDecision,
    WorkflowState,
    stable_digest,
)
from jevpilot.core import RoutingIntent

FIXTURE_DIR = Path(__file__).parent / "fixtures"
NoActionIntent = Literal["finish", "ask_human", "idle"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expected(_Model):
    """Acceptable routing behaviour for one decision case.

    A decision is *acceptable* if it invokes one of ``valid_capabilities``
    with inputs containing ``required_inputs[capability]`` (subset match),
    or proposes no action with an intent listed in ``no_action_intents``.
    """

    valid_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    unnecessary_capabilities: tuple[str, ...] = ()
    required_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    no_action_intents: tuple[NoActionIntent, ...] = ()


class HistoryItem(_Model):
    capability_id: str | None
    inputs: dict[str, Any] = Field(default_factory=dict)
    intent: NoActionIntent | Literal["invoke"] | None = None
    success: bool | None = None
    result: Any = None
    error: ErrorInfo | None = None
    control: ControlAction | None = None


class EvaluationItem(_Model):
    recommendation: ControlAction = ControlAction.CONTINUE
    goal_progress: float | None = None
    remaining_gaps: tuple[str, ...] = ()
    rationale: str = ""


class ArtifactItem(_Model):
    id: str | None = None
    name: str
    kind: str = "object"
    content: Any = None


class StateFixture(_Model):
    goal: Goal
    context: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactItem, ...] = ()
    history: tuple[HistoryItem, ...] = ()
    evaluations: tuple[EvaluationItem, ...] = ()


class DecisionCase(_Model):
    case_id: str
    pattern: str
    description: str = ""
    state: StateFixture
    capabilities: tuple[str | dict[str, Any], ...]
    expected: Expected


class DecisionFixtureFile(_Model):
    benchmark: str
    version: str
    suite: str | None = None  # suite whose rules/policy the baseline routers use
    capability_catalog: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cases: tuple[DecisionCase, ...]


class WorkflowCase(_Model):
    case_id: str
    suite: str
    pattern: str
    params: dict[str, Any] = Field(default_factory=dict)
    expected_status: str = "succeeded"
    optimal_steps: int | None = None
    unnecessary_capabilities: tuple[str, ...] = ()


class WorkflowFixtureFile(_Model):
    benchmark: str
    version: str
    cases: tuple[WorkflowCase, ...]


# -- loading --------------------------------------------------------------------


def load_decision_fixtures(path: Path | None = None) -> DecisionFixtureFile:
    return DecisionFixtureFile.model_validate(_read(path or FIXTURE_DIR / "decision_cases.json"))


def load_workflow_fixtures(path: Path | None = None) -> WorkflowFixtureFile:
    return WorkflowFixtureFile.model_validate(_read(path or FIXTURE_DIR / "workflow_cases.json"))


def fixture_digest(path: Path) -> str:
    return stable_digest(_read(path))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# -- building router inputs ---------------------------------------------------


def build_capabilities(
    case: DecisionCase, catalog: dict[str, dict[str, Any]]
) -> list[CapabilitySpec]:
    specs = []
    for entry in case.capabilities:
        if isinstance(entry, str):
            if entry not in catalog:
                raise KeyError(f"{case.case_id}: capability {entry!r} not in catalog")
            spec = {"id": entry, **catalog[entry]}
        else:
            spec = dict(entry)
        specs.append(spec_from_json(spec))
    return specs


def spec_from_json(data: dict[str, Any]) -> CapabilitySpec:
    """A :class:`CapabilitySpec` from a JSON description (JSON Schema for inputs)."""
    data = dict(data)
    schema = data.pop("input_schema", None)
    output = data.pop("output_schema", None)
    cid = data["id"]
    data.setdefault("name", cid.rsplit(".", 1)[-1])
    return CapabilitySpec(
        input_schema=model_from_json_schema(_model_name(cid, "Input"), schema) if schema else None,
        output_schema=model_from_json_schema(_model_name(cid, "Output"), output)
        if output
        else None,
        **data,
    )


_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


def model_from_json_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Pydantic model for a flat JSON Schema object.

    Supports ``properties`` with ``type`` (string, integer, number, boolean,
    array, object), ``enum``, ``minimum``/``maximum``, ``default``, plus
    ``required`` and ``additionalProperties: false``. That is enough for
    fixture capabilities. Nested objects are accepted but not checked.
    """
    if schema.get("type", "object") != "object":
        raise ValueError(f"{name}: input schema must describe an object")
    required = set(schema.get("required", ()))
    fields: dict[str, Any] = {}
    for key, prop in schema.get("properties", {}).items():
        if "enum" in prop:
            annotation: Any = Literal[tuple(prop["enum"])]
        else:
            annotation = _TYPES.get(prop.get("type", ""), Any)
        constraints = {
            k: v
            for k, v in (("ge", prop.get("minimum")), ("le", prop.get("maximum")))
            if v is not None
        }
        if key in required:
            fields[key] = (annotation, Field(..., **constraints))
        else:
            fields[key] = (annotation | None, Field(prop.get("default"), **constraints))
    extra: Literal["forbid", "ignore"] = (
        "forbid" if schema.get("additionalProperties") is False else "ignore"
    )
    model: type[BaseModel] = create_model(name, __config__=ConfigDict(extra=extra), **fields)
    return model


def _model_name(cid: str, suffix: str) -> str:
    return (
        "".join(part.title() for part in cid.replace("-", "_").replace(".", "_").split("_"))
        + suffix
    )


def build_state(case: DecisionCase) -> WorkflowState:
    """A real :class:`WorkflowState` reproducing the case's compact state description."""
    fx = case.state
    workflow_id = f"wf_fixture_{case.case_id}"
    artifacts = tuple(
        Artifact(
            name=a.name,
            kind=a.kind,
            content=a.content,
            provenance=Provenance(workflow_id=workflow_id),
            **({"id": a.id} if a.id else {}),
        )
        for a in fx.artifacts
    )
    observations: list[Observation] = []
    history: list[HistoryEntry] = []
    step = 0
    for i, item in enumerate(fx.history):
        intent = RoutingIntent(item.intent) if item.intent else None
        decision = RoutingDecision(
            id=f"dec_fixture_{i}",
            capability_id=item.capability_id,
            inputs=item.inputs,
            intent=intent,
            router_id="fixture",
        )
        control = ControlDecision(action=item.control, source="fixture") if item.control else None
        if item.capability_id is None or item.success is None:
            history.append(HistoryEntry(step=step, decision=decision, control=control))
            continue
        obs = Observation(
            id=f"obs_fixture_{i}",
            workflow_id=workflow_id,
            step=step,
            capability_id=item.capability_id,
            decision_id=decision.id,
            success=item.success,
            result=item.result,
            error=item.error,
            provenance=Provenance(
                workflow_id=workflow_id, step=step, capability_id=item.capability_id
            ),
        )
        observations.append(obs)
        history.append(
            HistoryEntry(step=step, decision=decision, observation_id=obs.id, control=control)
        )
        step += 1
    evaluations = tuple(
        EvaluationResult(
            evaluator_id="fixture",
            step=step,
            recommendation=e.recommendation,
            goal_progress=e.goal_progress,
            remaining_gaps=e.remaining_gaps,
            rationale=e.rationale,
        )
        for e in fx.evaluations
    )
    return WorkflowState(
        workflow_id=workflow_id,
        goal=fx.goal,
        context=fx.context,
        artifacts=artifacts,
        observations=tuple(observations),
        evaluations=evaluations,
        history=tuple(history),
        step=step,
        status="running",
    )
