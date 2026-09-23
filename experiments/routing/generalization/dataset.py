"""Benchmark splits, decision-case derivation and perturbations. BENCHMARK-ONLY.

Workflow cases live in ``benchmarks/routing/<split>/workflows.json``. Decision
cases are *derived* from them. The controller runs each workflow with a
recording oracle router that follows one reference trajectory, and every
state the controller presents to the router becomes one decision case. States
after failed observations are therefore exactly what a live workflow produces.

**Unseen composition** (``unseen_composition: true`` on an eval case) means,
and is verified by :func:`unseen_violations`, that:

1. every capability on the case's reference plan also appears on the reference
   plan of at least one dev case (the parts are known);
2. the case's goal fact set appears in no dev or validation case (the task is new);
3. at least one ordered transition ``A → B`` on its reference plan appears on
   no dev or validation reference plan (the composition is new).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from experiments.routing.generalization.oracle import (
    Expectation,
    Problem,
    expect,
    param_names,
    reference_plan,
)
from experiments.routing.generalization.world import (
    BENCHMARK_DIR,
    PERMANENT_MARKER,
    CapabilityDef,
    Catalog,
    ToolkitDomain,
    facts_of,
    initial_state,
)
from jevpilot import (
    CapabilitySpec,
    Controller,
    DefaultControlPolicy,
    HistoryEntry,
    Router,
    RoutingDecision,
    Runtime,
    WorkflowState,
    stable_digest,
)

SPLITS = ("dev", "validation", "eval")
Split = Literal["dev", "validation", "eval"]


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    level: int = Field(ge=1, le=5)
    categories: tuple[str, ...] = ()
    goal: str
    goal_facts: tuple[str, ...]
    forbidden_facts: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    initial_facts: tuple[str, ...] = ()
    capabilities: str | tuple[str, ...] = "core"
    extra_distractors: int = 0
    failures: dict[str, dict[str, Any]] = Field(default_factory=dict)
    expected_outcome: Literal["succeeded", "awaiting_human"] = "succeeded"
    unseen_composition: bool = False
    note: str = ""


def load_split(split: str, directory: Path = BENCHMARK_DIR) -> list[WorkflowSpec]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    data = json.loads((directory / split / "workflows.json").read_text(encoding="utf-8"))
    return [WorkflowSpec.model_validate(c) for c in data["cases"]]


def load_sample(name: str, directory: Path = BENCHMARK_DIR) -> dict[str, Any]:
    """A named, versioned subset of decision cases (``samples/<name>.json``)."""
    data: dict[str, Any] = json.loads(
        (directory / "samples" / f"{name}.json").read_text(encoding="utf-8")
    )
    return data


def split_digest(split: str, directory: Path = BENCHMARK_DIR) -> str:
    return stable_digest(json.loads((directory / split / "workflows.json").read_text()))


def seeded(*parts: Any) -> random.Random:
    return random.Random(int(stable_digest(list(parts))[:12], 16))


# -- problem construction ---------------------------------------------------------


def offered_ids(spec: WorkflowSpec, catalog: Catalog) -> list[str]:
    """Capabilities offered in this case, in their canonical presentation order."""
    if isinstance(spec.capabilities, tuple):
        ids = list(spec.capabilities)
    elif spec.capabilities == "core":
        ids = list(catalog.capabilities)
    elif spec.capabilities.startswith("core-without:"):
        drop = set(spec.capabilities.split(":", 1)[1].split(","))
        ids = [c for c in catalog.capabilities if c not in drop]
    else:
        raise ValueError(f"{spec.case_id}: unknown capability set {spec.capabilities!r}")
    if spec.extra_distractors:
        rng = seeded("distractors", spec.case_id)
        extra = rng.sample(sorted(catalog.distractors), spec.extra_distractors)
        for cid in extra:
            ids.insert(rng.randrange(len(ids) + 1), cid)
    return ids


def problem(spec: WorkflowSpec, catalog: Catalog, offered: Sequence[str]) -> Problem:
    return Problem(
        offered=tuple(catalog.get(c) for c in offered),
        goal=frozenset(spec.goal_facts),
        forbidden=frozenset(spec.forbidden_facts),
        known_params=frozenset(spec.parameters),
    )


def excluded_from(
    state: WorkflowState, canonical: Mapping[str, str] | None = None
) -> frozenset[str]:
    """Capabilities the router has *observed* to be permanently unavailable."""
    canonical = canonical or {}
    return frozenset(
        canonical.get(o.capability_id or "", o.capability_id or "")
        for o in state.observations
        if not o.success and o.error is not None and PERMANENT_MARKER in o.error.message
    )


def expectation_for(
    state: WorkflowState, p: Problem, catalog: Catalog, canonical: Mapping[str, str] | None = None
) -> Expectation:
    return expect(
        facts_of(state),
        p,
        excluded=excluded_from(state, canonical),
        all_params=param_names(catalog.all),
    )


def reference_steps(spec: WorkflowSpec, catalog: Catalog) -> int | None:
    """Lower bound on executed steps for a successful run: the shortest plan length."""
    offered = offered_ids(spec, catalog)
    permanent = frozenset(k for k, v in spec.failures.items() if v.get("mode") == "permanent")
    plan = reference_plan(frozenset(spec.initial_facts), problem(spec, catalog, offered), permanent)
    if spec.expected_outcome != "succeeded":
        return None
    return len(plan)


def plan_of(spec: WorkflowSpec, catalog: Catalog) -> list[str]:
    offered = offered_ids(spec, catalog)
    permanent = frozenset(k for k, v in spec.failures.items() if v.get("mode") == "permanent")
    return reference_plan(frozenset(spec.initial_facts), problem(spec, catalog, offered), permanent)


def bind_inputs(d: CapabilityDef, params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in d.inputs.items():
        param = spec.get("bind")
        if param and param in params:
            out[name] = params[param]
        elif "enum" in spec:
            out[name] = spec["enum"][0]
    return out


# -- decision-case derivation --------------------------------------------------------


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    workflow: WorkflowSpec
    split: str
    step_index: int
    state: WorkflowState
    offered: tuple[str, ...]  # canonical ids, presentation order
    expectation: Expectation


class _OracleRouter(Router):
    """Follows the oracle's preferred action and records every state it is shown."""

    router_id = "oracle"

    def __init__(self, spec: WorkflowSpec, catalog: Catalog, p: Problem) -> None:
        self.spec, self.catalog, self.p = spec, catalog, p
        self.seen: list[tuple[WorkflowState, Expectation]] = []

    def select_next(
        self, state: WorkflowState, capabilities: Sequence[CapabilitySpec]
    ) -> RoutingDecision:
        e = expectation_for(state, self.p, self.catalog)
        self.seen.append((state, e))
        if e.kind == "finish":
            return RoutingDecision.finish("oracle: goal met", router_id=self.router_id)
        if e.kind == "ask_human" and e.acceptable:
            cid = sorted(e.acceptable)[0]  # progress that needs no missing information, then ask
        elif e.kind != "invoke" or not e.preferred:
            return RoutingDecision.ask_human("oracle: blocked", router_id=self.router_id)
        else:
            cid = sorted(e.preferred)[0]
        return RoutingDecision(
            capability_id=cid,
            inputs=bind_inputs(self.catalog.get(cid), self.spec.parameters),
            router_id=self.router_id,
        )


def build_controller(
    spec: WorkflowSpec,
    catalog: Catalog,
    router: Router,
    offered: Sequence[str],
    *,
    rename: Mapping[str, str] | None = None,
    max_steps: int | None = None,
    trace_sinks: Sequence[Any] = (),
) -> Controller:
    domain = ToolkitDomain(
        catalog,
        offered,
        goal_facts=frozenset(spec.goal_facts),
        forbidden=frozenset(spec.forbidden_facts),
        failures=spec.failures,
        rename=rename,
    )
    runtime = Runtime()
    runtime.load(domain)
    ref = reference_steps(spec, catalog) or 0
    policy = DefaultControlPolicy(
        max_steps=max_steps or max(8, 2 * ref + 4),
        max_retries=0,  # failures go back to the router: recovery is what is being measured
        max_consecutive_failures=4,
        max_routing_failures=0,
    )
    return Controller(
        runtime.capabilities,
        router,
        domain.evaluators()[0],
        policy=policy,
        trace_sinks=trace_sinks,
        snapshot_states=False,
    )


def start_state(spec: WorkflowSpec, split: str) -> WorkflowState:
    return initial_state(
        spec.goal, spec.parameters, spec.initial_facts, workflow_id=f"wf_{split}_{spec.case_id}"
    )


def decision_cases(split: str, catalog: Catalog) -> list[DecisionCase]:
    cases: list[DecisionCase] = []
    for spec in load_split(split):
        offered = offered_ids(spec, catalog)
        p = problem(spec, catalog, offered)
        oracle = _OracleRouter(spec, catalog, p)
        build_controller(spec, catalog, oracle, offered).run(start_state(spec, split))
        seen: set[str] = set()
        for i, (state, e) in enumerate(oracle.seen):
            key = stable_digest([sorted(facts_of(state)), [o.success for o in state.observations]])
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                DecisionCase(f"{spec.case_id}@{i}", spec, split, i, state, tuple(offered), e)
            )
    return cases


# -- perturbations ----------------------------------------------------------------


@dataclass(frozen=True)
class Presentation:
    """What routers are shown for one decision: renamed state and ordered specs."""

    label: str
    state: WorkflowState
    specs: tuple[CapabilitySpec, ...]
    offered: tuple[str, ...]  # canonical ids in presented order
    to_canonical: Mapping[str, str]
    expectation: Expectation


def present(
    case: DecisionCase,
    catalog: Catalog,
    *,
    label: str = "main",
    order: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    expectation: Expectation | None = None,
) -> Presentation:
    ids = tuple(order or case.offered)
    rename = dict(rename or {})
    specs = tuple(
        catalog.get(c).spec(cap_id=rename[c], name=rename[c])
        if c in rename
        else catalog.get(c).spec()
        for c in ids
    )
    return Presentation(
        label,
        rename_state(case.state, rename),
        specs,
        ids,
        {v: k for k, v in rename.items()},
        expectation or case.expectation,
    )


def permuted(case: DecisionCase, catalog: Catalog, k: int, seed: int) -> Presentation:
    order = list(case.offered)
    seeded("order", seed, case.case_id, k).shuffle(order)
    return present(case, catalog, label=f"order:{k}", order=order)


def opaque_names(ids: Iterable[str], seed: int, key: str) -> dict[str, str]:
    ids = sorted(ids)
    numbers = list(range(1, len(ids) + 1))
    seeded("names", seed, key).shuffle(numbers)
    return {cid: f"cap_{n:02d}" for cid, n in zip(ids, numbers, strict=True)}


def renamed(case: DecisionCase, catalog: Catalog, seed: int) -> Presentation:
    return present(
        case, catalog, label="names", rename=opaque_names(case.offered, seed, case.workflow.case_id)
    )


def with_distractors(
    case: DecisionCase, catalog: Catalog, total: int, seed: int
) -> Presentation | None:
    """``total`` offered capabilities: the goal-relevant ones plus irrelevant fill."""
    relevant = relevant_ids(case.workflow, catalog, case.offered)
    if len(relevant) > total:
        return None
    pool = sorted(set(catalog.all) - set(relevant))
    rng = seeded("scale", seed, case.case_id, total)
    ids = relevant + rng.sample(pool, total - len(relevant))
    rng.shuffle(ids)
    p = problem(case.workflow, catalog, ids)
    return present(
        case,
        catalog,
        label=f"distractors:{total}",
        order=ids,
        expectation=expectation_for(case.state, p, catalog),
    )


def relevant_ids(spec: WorkflowSpec, catalog: Catalog, offered: Sequence[str]) -> list[str]:
    """Offered capabilities that produce a fact the goal can depend on (plus forbidden traps)."""
    defs = [catalog.get(c) for c in offered]
    relevant = set(spec.goal_facts)
    changed = True
    while changed:
        changed = False
        for d in defs:
            if d.produces & relevant and not d.requires <= relevant:
                relevant |= d.requires
                changed = True
    forbidden = set(spec.forbidden_facts)
    return [d.id for d in defs if d.produces & (relevant | forbidden)]


def rename_state(state: WorkflowState, rename: Mapping[str, str]) -> WorkflowState:
    if not rename:
        return state

    def r(cid: str | None) -> str | None:
        return rename.get(cid, cid) if cid else cid

    observations = tuple(
        o.model_copy(update={"capability_id": r(o.capability_id)}) for o in state.observations
    )
    history = tuple(
        HistoryEntry(
            step=h.step,
            decision=h.decision.model_copy(update={"capability_id": r(h.decision.capability_id)}),
            observation_id=h.observation_id,
            control=h.control,
            routing_error=h.routing_error,
            timestamp=h.timestamp,
        )
        for h in state.history
    )
    return state.evolve(observations=observations, history=history)


# -- unseen-composition verification -------------------------------------------------


def _bigrams(plan: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(plan, plan[1:], strict=False))


def unseen_violations(catalog: Catalog) -> list[str]:
    """Why eval cases tagged ``unseen_composition`` fail the definition (empty: all fine)."""
    dev = load_split("dev")
    seen_specs = dev + load_split("validation")
    dev_caps = {c for s in dev for c in plan_of(s, catalog)}
    seen_goals = {frozenset(s.goal_facts) for s in seen_specs}
    seen_bigrams = set().union(*(_bigrams(plan_of(s, catalog)) for s in seen_specs))
    problems = []
    for s in load_split("eval"):
        if not s.unseen_composition:
            continue
        plan = plan_of(s, catalog)
        unknown = set(plan) - dev_caps
        if unknown:
            problems.append(f"{s.case_id}: capabilities never used in dev: {sorted(unknown)}")
        if frozenset(s.goal_facts) in seen_goals:
            problems.append(f"{s.case_id}: goal {sorted(s.goal_facts)} appears in dev/validation")
        if not _bigrams(plan) - seen_bigrams:
            problems.append(f"{s.case_id}: every transition of {plan} was already demonstrated")
    return problems
