"""Run generalization experiments and produce per-decision / per-workflow records.

Experiments (all routers see exactly the same state and capabilities per item):

``main``         every decision case and every workflow case of the split
``order``        decision cases under K seeded permutations of capability order
``names``        decision cases with capability ids/names replaced by opaque ``cap_NN``
``distractors``  decision + workflow cases with N total capabilities (relevant + filler)
``fallback``     workflow cases run by fallback *systems* (e.g. ``jev>rule``)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.oracle import Expectation, expect, param_names
from experiments.routing.generalization.routers import RouterSlot
from experiments.routing.generalization.world import PERMANENT_MARKER, Catalog
from jevpilot import (
    JsonlTraceSink,
    Router,
    RoutingDecision,
    RoutingRequest,
    TraceEvent,
    TraceEventType,
    stable_digest,
    validate_decision,
)
from jevpilot.exceptions import RoutingError

Record = dict[str, Any]
USAGE = ("model_calls", "input_tokens", "output_tokens", "estimated_cost_usd")
EXPERIMENTS = ("main", "order", "names", "distractors", "fallback")


@dataclass
class RunConfig:
    split: str = "eval"
    experiments: Sequence[str] = ("main",)
    repetitions: int = 1
    seed: int = 0
    order_permutations: int = 3
    distractor_levels: Sequence[int] = (8, 16, 32, 48)
    kinds: Sequence[str] = ("decision", "workflow")
    trace_dir: Path | None = None
    progress: Callable[[str], None] | None = None
    decision_case_ids: frozenset[str] | None = None  # restrict to a named sample


@dataclass
class Results:
    decisions: list[Record] = field(default_factory=list)
    workflows: list[Record] = field(default_factory=list)
    requests: dict[str, Any] = field(default_factory=dict)


def item_seed(*parts: Any) -> int:
    return int(stable_digest(list(parts))[:8], 16)


# -- decisions -----------------------------------------------------------------


def score(
    decision: RoutingDecision | None, canonical: str | None, e: Expectation, inputs_ok: bool | None
) -> dict[str, Any]:
    if decision is None:
        return {"acceptable": False, "preferred": False, "forbidden": False, "unnecessary": False}
    if canonical is None:
        intent = str(decision.intent)
        best = {"finish": "finish", "ask_human": "ask_human", "impossible": "ask_human"}
        return {
            "acceptable": intent in e.no_action_intents,
            "preferred": best.get(e.kind) == intent,
            "forbidden": False,
            "unnecessary": False,
        }
    return {
        "acceptable": canonical in e.acceptable and bool(inputs_ok),
        "preferred": canonical in e.preferred and bool(inputs_ok),
        "forbidden": canonical in e.forbidden,
        "unnecessary": canonical in e.unnecessary,
    }


def inputs_correct(
    catalog: Catalog, canonical: str, inputs: Mapping[str, Any], params: Mapping[str, Any]
) -> bool:
    d = catalog.get(canonical)
    return all(inputs.get(f) == params[p] for f, p in d.bindings.items() if p in params)


def decide(
    router: Router,
    pres: ds.Presentation,
    case: ds.DecisionCase,
    catalog: Catalog,
    results: Results,
    base: Record,
) -> Record:
    request = RoutingRequest.from_state(pres.state, pres.specs)
    fingerprint = request.fingerprint()
    results.requests.setdefault(fingerprint, request.to_dict())
    decision: RoutingDecision | None = None
    error: str | None = None
    attempts: list[Any] = []
    t0 = time.perf_counter()
    try:
        outcome = router.route(pres.state, pres.specs)
        validate_decision(outcome.decision, pres.specs)  # exactly as the controller does
        decision, attempts = outcome.decision, list(outcome.attempts)
    except RoutingError as exc:
        error = type(exc).__name__
        attempts = list(exc.attempts)
    latency_ms = (time.perf_counter() - t0) * 1000
    shown = decision.capability_id if decision else None
    canonical = pres.to_canonical.get(shown, shown) if shown else None
    ok = (
        inputs_correct(catalog, canonical, decision.inputs, case.workflow.parameters)
        if decision is not None and canonical
        else None
    )
    e = pres.expectation
    return {
        **base,
        "case_id": case.case_id,
        "workflow_case": case.workflow.case_id,
        "step_index": case.step_index,
        "level": case.workflow.level,
        "categories": list(case.workflow.categories),
        "unseen": case.workflow.unseen_composition,
        "expected_kind": e.kind,
        "oracle_acceptable": sorted(e.acceptable),
        "oracle_preferred": sorted(e.preferred),
        "oracle_no_action": sorted(e.no_action_intents),
        "n_capabilities": len(pres.specs),
        "state_facts": sorted(pres.state.context["facts"]),
        "request_fingerprint": fingerprint,
        "valid": decision is not None,
        "error_type": error,
        "capability_id": shown,
        "canonical_capability_id": canonical,
        "intent": str(decision.intent) if decision else None,
        "inputs": decision.inputs if decision else None,
        "inputs_ok": ok,
        **score(decision, canonical, e, ok),
        "confidence": decision.confidence if decision else None,
        "routing_latency_ms": latency_ms,
        **attempt_summary(a.model_dump(mode="json") for a in attempts),
    }


def attempt_summary(attempts: Any) -> Record:
    attempts = list(attempts)
    meta = [a.get("metadata") or {} for a in attempts]
    out: Record = {
        "attempts": len(attempts),
        "attempt_routers": [a["router_id"] for a in attempts],
        "attempt_errors": [a["error"]["type"] for a in attempts if a.get("error")],
        "fallback_used": any(not a["success"] for a in attempts)
        and any(a["success"] for a in attempts),
        "providers": sorted({m["provider"] for m in meta if m.get("provider")}),
        "requested_models": sorted(
            {m["requested_model"] for m in meta if m.get("requested_model")}
        ),
        "actual_models": sorted({a["model"] for a in attempts if a.get("model")}),
    }
    for key in USAGE:
        known = [
            a["usage"][key] for a in attempts if a.get("usage") and a["usage"].get(key) is not None
        ]
        out[key] = sum(known) if known else None
    return out


# -- workflows ------------------------------------------------------------------


def run_workflow(
    router: Router,
    spec: ds.WorkflowSpec,
    catalog: Catalog,
    offered: Sequence[str],
    split: str,
    base: Record,
    trace_file: Path | None,
) -> Record:
    sinks = [JsonlTraceSink(trace_file)] if trace_file else []
    controller = ds.build_controller(spec, catalog, router, offered, trace_sinks=sinks)
    t0 = time.perf_counter()
    result = controller.run(ds.start_state(spec, split))
    total_ms = (time.perf_counter() - t0) * 1000
    state = result.state
    facts = set(state.context.get("facts", ()))
    p = ds.problem(spec, catalog, offered)
    ref = ds.reference_steps(spec, catalog)
    status = str(state.status)
    forbidden = bool(facts & set(spec.forbidden_facts))
    completed = status == spec.expected_outcome and not forbidden
    calls = judge_calls(state, spec, catalog, p)
    last = state.history[-1].decision if state.history else None
    return {
        **base,
        "case_id": spec.case_id,
        "level": spec.level,
        "categories": list(spec.categories),
        "unseen": spec.unseen_composition,
        "expected_outcome": spec.expected_outcome,
        "n_capabilities": len(offered),
        "status": status,
        "completed": completed,
        "forbidden_reached": forbidden,
        "steps": state.step,
        "reference_steps": ref,
        "excess_steps": state.step - ref if completed and ref is not None else None,
        "calls": len(calls),
        "unnecessary_calls": sum(not c["necessary"] for c in calls),
        "failed_calls": sum(not c["success"] for c in calls),
        "router_retries": sum(
            1
            for a, b in zip(calls, calls[1:], strict=False)
            if not a["success"] and a["capability"] == b["capability"]
        ),
        "final_intent": str(last.intent) if last else None,
        "asked_human": bool(last and str(last.intent) == "ask_human"),
        "capability_sequence": [c["capability"] for c in calls],
        "final_control": str(result.control.action),
        "final_reason": result.control.reason,
        "orchestration_error": result.error.type if result.error else None,
        "total_latency_ms": total_ms,
        "trace_file": str(trace_file) if trace_file else None,
        **trace_summary(result.trace),
    }


def judge_calls(state: Any, spec: ds.WorkflowSpec, catalog: Catalog, p: Any) -> list[Record]:
    """Replay the run and ask the oracle whether each executed call was on a shortest path."""
    facts = frozenset(spec.initial_facts)
    excluded: set[str] = set()
    all_params = param_names(catalog.all)
    out = []
    for obs in state.observations:
        cid = obs.capability_id or ""
        e = expect(facts, p, excluded=frozenset(excluded), all_params=all_params)
        out.append(
            {
                "capability": cid,
                "success": obs.success,
                "necessary": cid in e.acceptable and e.kind in ("invoke", "ask_human"),
            }
        )
        if obs.success:
            facts = facts | catalog.get(cid).produces
        elif obs.error is not None and PERMANENT_MARKER in obs.error.message:
            excluded.add(cid)
    return out


def trace_summary(trace: Sequence[TraceEvent]) -> Record:
    routing_ms: list[float] = []
    execution_ms: list[float] = []
    attempts: list[Any] = []
    failures = fallbacks = replans = decisions = 0
    errors: list[str] = []
    intents: list[str] = []
    for ev in trace:
        p = ev.payload
        if ev.type is TraceEventType.ROUTING_DECISION and "attempts" in p:
            decisions += 1
            routing_ms.append(p["routing_latency_s"] * 1000)
            fallbacks += bool(p.get("fallback_used"))
            attempts += p["attempts"]
            intents.append(p["decision"]["intent"])
        elif ev.type is TraceEventType.ROUTING_FAILURE:
            failures += 1
            routing_ms.append(p["routing_latency_s"] * 1000)
            errors.append(p["error"]["type"])
            attempts += p["attempts"]
        elif ev.type is TraceEventType.OBSERVATION:
            execution_ms.append(p["observation"]["execution_time"] * 1000)
        elif ev.type is TraceEventType.CONTROL_DECISION:
            replans += p["control"]["action"] == "replan"
    return {
        "routing_decisions": decisions,
        "routing_failures": failures,
        "routing_errors": errors,
        "fallback_decisions": fallbacks,
        "replans": replans,
        "decision_intents": intents,
        "routing_latencies_ms": routing_ms,
        "execution_latencies_ms": execution_ms,
        "routing_latency_ms": sum(routing_ms),
        "execution_latency_ms": sum(execution_ms),
        **attempt_summary(attempts),
    }


# -- orchestration ----------------------------------------------------------------


def run(
    slots: Sequence[RouterSlot],
    catalog: Catalog,
    cfg: RunConfig,
    systems: Sequence[RouterSlot] = (),
) -> Results:
    results = Results()
    cases = ds.decision_cases(cfg.split, catalog) if "decision" in cfg.kinds else []
    if cfg.decision_case_ids is not None:
        missing = cfg.decision_case_ids - {c.case_id for c in cases}
        if missing:
            raise ValueError(f"sample cases not found in split {cfg.split!r}: {sorted(missing)}")
        cases = [c for c in cases if c.case_id in cfg.decision_case_ids]
    specs = ds.load_split(cfg.split) if "workflow" in cfg.kinds else []
    say = cfg.progress or (lambda _msg: None)
    runnable = [s for s in slots if s.available]
    for rep in range(cfg.repetitions):
        for slot in runnable:
            base_meta = {
                "split": cfg.split,
                "router": slot.name,
                "router_kind": slot.kind,
                "repetition": rep,
            }
            for exp in cfg.experiments:
                if exp == "fallback":
                    continue
                say(f"rep {rep} · {slot.name} · {exp}")
                for label, case, pres in _decision_items(exp, cases, catalog, cfg):
                    seed = item_seed(cfg.seed, exp, label, case.case_id, rep)
                    base = {**base_meta, "experiment": exp, "perturbation": label, "seed": seed}
                    router = slot.factory(seed) if slot.factory else None
                    assert router is not None
                    results.decisions.append(decide(router, pres, case, catalog, results, base))
                for label, spec, offered in _workflow_items(exp, specs, catalog, cfg):
                    seed = item_seed(cfg.seed, exp, label, spec.case_id, rep)
                    base = {**base_meta, "experiment": exp, "perturbation": label, "seed": seed}
                    router = slot.factory(seed) if slot.factory else None
                    assert router is not None
                    results.workflows.append(
                        run_workflow(
                            router,
                            spec,
                            catalog,
                            offered,
                            cfg.split,
                            base,
                            _trace_path(cfg, slot.name, exp, label, spec.case_id, rep),
                        )
                    )
        if "fallback" in cfg.experiments:
            for system in (s for s in systems if s.available):
                say(f"rep {rep} · {system.name} · fallback")
                for spec in specs:
                    seed = item_seed(cfg.seed, "fallback", spec.case_id, rep)
                    base = {
                        "split": cfg.split,
                        "router": system.name,
                        "router_kind": "system",
                        "repetition": rep,
                        "experiment": "fallback",
                        "perturbation": "main",
                        "seed": seed,
                    }
                    assert system.factory is not None
                    results.workflows.append(
                        run_workflow(
                            system.factory(seed),
                            spec,
                            catalog,
                            ds.offered_ids(spec, catalog),
                            cfg.split,
                            base,
                            _trace_path(cfg, system.name, "fallback", "main", spec.case_id, rep),
                        )
                    )
    return results


def _decision_items(
    exp: str, cases: Sequence[ds.DecisionCase], catalog: Catalog, cfg: RunConfig
) -> Any:
    for case in cases:
        if exp == "main":
            yield "main", case, ds.present(case, catalog)
        elif exp == "order":
            for k in range(cfg.order_permutations):
                yield f"order:{k}", case, ds.permuted(case, catalog, k, cfg.seed)
        elif exp == "names":
            yield "names", case, ds.renamed(case, catalog, cfg.seed)
        elif exp == "distractors":
            for total in cfg.distractor_levels:
                pres = ds.with_distractors(case, catalog, total, cfg.seed)
                if pres is not None:
                    yield pres.label, case, pres


def _workflow_items(
    exp: str, specs: Sequence[ds.WorkflowSpec], catalog: Catalog, cfg: RunConfig
) -> Any:
    for spec in specs:
        offered = ds.offered_ids(spec, catalog)
        if exp == "main":
            yield "main", spec, offered
        elif exp == "distractors":
            relevant = ds.relevant_ids(spec, catalog, offered)
            for total in cfg.distractor_levels:
                if len(relevant) > total:
                    continue
                rng = ds.seeded("scale-wf", cfg.seed, spec.case_id, total)
                pool = sorted(set(catalog.all) - set(relevant))
                ids = relevant + rng.sample(pool, total - len(relevant))
                rng.shuffle(ids)
                yield f"distractors:{total}", spec, ids


def _trace_path(
    cfg: RunConfig, router: str, exp: str, label: str, case_id: str, rep: int
) -> Path | None:
    if cfg.trace_dir is None:
        return None
    safe = f"{exp}__{label.replace(':', '-')}__{case_id}__r{rep}.jsonl"
    return cfg.trace_dir / router.replace(">", "-then-") / safe
