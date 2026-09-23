"""The "analysis toolkit" world: an executable, domain-neutral benchmark environment.

State is a set of *facts* (``state.context["facts"]``). Each capability
requires some facts, adds others, and may take inputs, some of which must
match a goal parameter (``bind``). Routers see natural-language descriptions,
the preconditions and effects as text, JSON input schemas, the goal
*description*, the current facts and the full action history. The goal's
target facts are **not** shown to routers. The benchmark evaluator holds them
and defines success purely from the resulting state, so any valid ordering of
actions succeeds.

Precondition failures, bad inputs and injected failures surface as failed
observations (``CapabilityError``), exactly as a real tool failure would. The
router must read them and react.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.routing.cases import model_from_json_schema
from jevpilot import (
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ControlAction,
    DomainModule,
    EvaluationResult,
    Evaluator,
    ExecutionContext,
    Goal,
    StateEffects,
    WorkflowState,
)
from jevpilot.exceptions import CapabilityError

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "routing"
DOMAIN = "toolkit"
PERMANENT_MARKER = "permanently unavailable"


@dataclass(frozen=True, eq=False)  # identity hash: catalog entries are singletons
class CapabilityDef:
    """A catalog entry: the executable semantics behind one capability."""

    id: str
    name: str
    description: str
    requires: frozenset[str] = frozenset()
    produces: frozenset[str] = frozenset()
    inputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cost: float = 1.0
    side_effects: bool = False
    group: str = "core"

    @property
    def bindings(self) -> dict[str, str]:
        """Input field → goal parameter whose value it must equal."""
        return {k: v["bind"] for k, v in self.inputs.items() if "bind" in v}

    @property
    def required_params(self) -> frozenset[str]:
        """Goal parameters that must be known to use this capability."""
        return frozenset(
            v["bind"] for v in self.inputs.values() if "bind" in v and not v.get("optional_bind")
        )

    def input_schema(self) -> dict[str, Any]:
        props = {}
        for name, spec in self.inputs.items():
            prop = {k: v for k, v in spec.items() if k in ("type", "description", "enum")}
            props[name] = prop
        return {
            "type": "object",
            "properties": props,
            "required": sorted(self.inputs),
            "additionalProperties": False,
        }

    def spec(self, *, cap_id: str | None = None, name: str | None = None) -> CapabilitySpec:
        """Router-facing spec. ``cap_id``/``name`` override for the name-perturbation test."""
        cid = cap_id or self.id
        return CapabilitySpec(
            id=cid,
            name=name or self.name,
            description=self.description,
            domain=DOMAIN,
            input_schema=model_from_json_schema("In_" + cid.replace(".", "_"), self.input_schema()),
            preconditions=tuple(f"requires fact '{f}'" for f in sorted(self.requires)),
            effects=tuple(f"adds fact '{f}'" for f in sorted(self.produces)),
            side_effects=self.side_effects,
            cost_estimate=self.cost,
            tags=frozenset({self.group}),
        )


@dataclass(frozen=True)
class Catalog:
    capabilities: Mapping[str, CapabilityDef]
    distractors: Mapping[str, CapabilityDef]

    @property
    def all(self) -> dict[str, CapabilityDef]:
        return {**self.capabilities, **self.distractors}

    def get(self, cid: str) -> CapabilityDef:
        return self.all[cid]


def load_catalog(directory: Path = BENCHMARK_DIR) -> Catalog:
    def read(name: str, group: str) -> dict[str, CapabilityDef]:
        entries = json.loads((directory / name).read_text(encoding="utf-8"))["capabilities"]
        return {
            e["id"]: CapabilityDef(
                id=e["id"],
                name=e["name"],
                description=e["description"],
                requires=frozenset(e.get("requires", ())),
                produces=frozenset(e.get("produces", ())),
                inputs=e.get("inputs", {}),
                cost=float(e.get("cost", 1.0)),
                side_effects=bool(e.get("side_effects", False)),
                group=group,
            )
            for e in entries
        }

    return Catalog(read("catalog.json", "core"), read("distractors.json", "distractor"))


# -- execution ---------------------------------------------------------------------


def facts_of(state: WorkflowState) -> frozenset[str]:
    return frozenset(state.context.get("facts", ()))


class FactCapability(Capability):
    """Executes a :class:`CapabilityDef` against the facts in the workflow state."""

    def __init__(
        self,
        definition: CapabilityDef,
        spec: CapabilitySpec,
        failures: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.definition = definition
        self.spec = spec
        self._failure = failures.get(definition.id)

    def execute(self, inputs: Any, ctx: ExecutionContext) -> CapabilityResult:
        d = self.definition
        state = ctx.state
        facts = facts_of(state)
        missing = sorted(d.requires - facts)
        if missing:
            raise CapabilityError(f"precondition not met: missing facts {missing}", retryable=False)
        values = inputs.model_dump() if hasattr(inputs, "model_dump") else dict(inputs)
        params = state.goal.parameters
        for fld, param in d.bindings.items():
            optional = d.inputs[fld].get("optional_bind", False)
            if param not in params:
                if optional:
                    continue
                raise CapabilityError(
                    f"no {param!r} is known for this workflow; '{fld}'={values.get(fld)!r} "
                    "does not exist",
                    retryable=False,
                )
            if values.get(fld) != params[param]:
                raise CapabilityError(
                    f"'{fld}'={values.get(fld)!r} does not match any available {param}",
                    retryable=False,
                )
        if self._failure is not None:
            if self._failure["mode"] == "permanent":
                raise CapabilityError(f"service {PERMANENT_MARKER}", retryable=False)
            attempts = sum(1 for o in state.observations if o.capability_id == self.spec.id)
            if attempts < int(self._failure.get("times", 1)):
                raise CapabilityError("temporary failure: service busy, try again", retryable=True)
        added = sorted(d.produces - facts)
        return CapabilityResult(
            output={"added_facts": added},
            effects=StateEffects(context_updates={"facts": sorted(facts | d.produces)}),
        )


class FactGoalEvaluator(Evaluator):
    """Success iff every goal fact holds; failure iff a forbidden fact appears.

    The evaluator belongs to the benchmark, so routers never see the goal facts.
    """

    evaluator_id = "toolkit.goal_facts"

    def __init__(self, goal_facts: frozenset[str], forbidden: frozenset[str]) -> None:
        self.goal_facts = goal_facts
        self.forbidden = forbidden

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        facts = facts_of(state)
        bad = sorted(facts & self.forbidden)
        if bad:
            return EvaluationResult(
                evaluator_id=self.evaluator_id,
                step=state.step,
                goal_progress=0.0,
                recommendation=ControlAction.TERMINATE_FAILURE,
                rationale=f"forbidden outcome reached: {bad}",
            )
        done = self.goal_facts & facts
        if done == self.goal_facts:
            return EvaluationResult(
                evaluator_id=self.evaluator_id,
                step=state.step,
                goal_progress=1.0,
                recommendation=ControlAction.TERMINATE_SUCCESS,
                rationale="goal satisfied",
            )
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            step=state.step,
            goal_progress=len(done) / max(len(self.goal_facts), 1),
            remaining_gaps=("goal not yet satisfied",),
            rationale="goal not yet satisfied",
        )


class ToolkitDomain(DomainModule):
    """One workflow instance: the offered capabilities (possibly renamed) and its evaluator."""

    name = DOMAIN
    version = "1.0.0"
    description = "Artificial analysis-toolkit world for routing generalization benchmarks."

    def __init__(
        self,
        catalog: Catalog,
        offered: Sequence[str],
        *,
        goal_facts: frozenset[str],
        forbidden: frozenset[str],
        failures: Mapping[str, Mapping[str, Any]] | None = None,
        rename: Mapping[str, str] | None = None,
    ) -> None:
        self.catalog = catalog
        self.offered = list(offered)
        self.rename = dict(rename or {})
        self._evaluator = FactGoalEvaluator(goal_facts, forbidden)
        self._failures = dict(failures or {})

    def capabilities(self) -> Sequence[Capability]:
        out: list[Capability] = []
        for cid in self.offered:
            d = self.catalog.get(cid)
            alias = self.rename.get(cid)
            spec = d.spec(cap_id=alias, name=alias) if alias else d.spec()
            out.append(FactCapability(d, spec, self._failures))
        return out

    def evaluators(self) -> Sequence[Evaluator]:
        return [self._evaluator]


def initial_state(
    goal: str, parameters: Mapping[str, Any], facts: Sequence[str], workflow_id: str
) -> WorkflowState:
    return WorkflowState(
        workflow_id=workflow_id,
        goal=Goal(description=goal, parameters=dict(parameters)),
        context={"facts": sorted(facts)},
        metadata={"domain": DOMAIN},
    )
