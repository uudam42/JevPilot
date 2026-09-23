"""Metrics for generalization runs.

Every metric is computed **per repetition** and then aggregated as mean,
median, std, min, max and the per-repetition values. Pooled numerators and
denominators are kept, so small samples stay visible. Token usage and cost
stay ``None`` unless a provider reported them.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

Record = dict[str, Any]
Metric = Callable[[list[Record]], tuple[int, int] | float | None]


def ratio(num: int, den: int) -> dict[str, Any]:
    return {"value": num / den if den else None, "numerator": num, "denominator": den}


def spread(values: Sequence[float | None]) -> dict[str, Any]:
    vals: list[float] = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def by_rep(
    records: list[Record], metric: Callable[[list[Record]], tuple[int, int]]
) -> dict[str, Any]:
    """Pooled ratio plus its distribution across repetitions."""
    groups: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        groups[r["repetition"]].append(r)
    num = den = 0
    per_rep = []
    for rep in sorted(groups):
        n, d = metric(groups[rep])
        num, den = num + n, den + d
        per_rep.append(n / d if d else None)
    return {
        **ratio(num, den),
        "across_repetitions": spread(per_rep),
        "per_repetition": per_rep,
        "success_count": num,
    }


def _count(
    pred: Callable[[Record], bool], within: Callable[[Record], bool] = lambda r: True
) -> Callable[[list[Record]], tuple[int, int]]:
    def metric(rs: list[Record]) -> tuple[int, int]:
        pool = [r for r in rs if within(r)]
        return sum(1 for r in pool if pred(r)), len(pool)

    return metric


def usage(records: Iterable[Record]) -> dict[str, Any]:
    records = list(records)
    out: dict[str, Any] = {}
    for key in ("model_calls", "input_tokens", "output_tokens", "estimated_cost_usd"):
        known = [r[key] for r in records if r.get(key) is not None]
        out[key] = sum(known) if known else None
    out["total_tokens"] = (
        (out["input_tokens"] or 0) + (out["output_tokens"] or 0)
        if out["input_tokens"] is not None or out["output_tokens"] is not None
        else None
    )
    return out


# -- decision level -----------------------------------------------------------------


def _asks(r: Record) -> bool:
    return r.get("intent") == "ask_human"


def _should_ask(r: Record) -> bool:
    return r["expected_kind"] in ("ask_human", "impossible")


def decision_metrics(rs: list[Record]) -> dict[str, Any]:
    attempts = [(r, e) for r in rs for e in r["attempt_errors"]]
    total_attempts = sum(r["attempts"] for r in rs)
    conf = [
        (r["confidence"], r["acceptable"]) for r in rs if r["valid"] and r["confidence"] is not None
    ]
    return {
        "decisions": len(rs),
        "valid_decision_rate": by_rep(rs, _count(lambda r: r["valid"])),
        "routing_accuracy": by_rep(rs, _count(lambda r: r["acceptable"])),
        "preferred_rate": by_rep(rs, _count(lambda r: r["preferred"])),
        "invalid_capability_rate": {
            **ratio(sum(e == "InvalidCapabilityError" for _, e in attempts), total_attempts),
            "note": "per router attempt",
        },
        "invalid_input_rate": ratio(
            sum(e == "InvalidRoutingInputError" for _, e in attempts), total_attempts
        ),
        "malformed_rate": ratio(
            sum(e == "MalformedRoutingDecisionError" for _, e in attempts), total_attempts
        ),
        "adapter_failure_rate": ratio(
            sum(e in ("RouterAdapterError", "RouterTimeoutError") for _, e in attempts),
            total_attempts,
        ),
        "forbidden_selection_rate": by_rep(rs, _count(lambda r: r["forbidden"])),
        "unnecessary_selection_rate": by_rep(rs, _count(lambda r: r["unnecessary"])),
        "ask_human": {
            "recall": by_rep(rs, _count(_asks, within=_should_ask)),
            "precision": by_rep(rs, _count(_should_ask, within=_asks)),
            "inappropriate": by_rep(rs, _count(_asks, within=lambda r: not _should_ask(r))),
            # ask_human OR idle: both end the run in AWAITING_HUMAN via the controller
            "appropriate_stop": by_rep(
                rs, _count(lambda r: r.get("intent") in ("ask_human", "idle"), within=_should_ask)
            ),
        },
        "finish": {
            "recall": by_rep(
                rs,
                _count(
                    lambda r: r.get("intent") in ("finish", "idle"),
                    within=lambda r: r["expected_kind"] == "finish",
                ),
            ),
            "explicit_finish_recall": by_rep(
                rs,
                _count(
                    lambda r: r.get("intent") == "finish",
                    within=lambda r: r["expected_kind"] == "finish",
                ),
            ),
            "premature": by_rep(
                rs,
                _count(
                    lambda r: r.get("intent") == "finish",
                    within=lambda r: r["expected_kind"] == "invoke",
                ),
            ),
        },
        "routing_latency_ms": spread([r["routing_latency_ms"] for r in rs]),
        "confidence_vs_correctness": {
            "reported": len(conf),
            "mean_when_acceptable": _mean([c for c, ok in conf if ok]),
            "mean_when_not_acceptable": _mean([c for c, ok in conf if not ok]),
        },
        "usage": usage(rs),
    }


def breakdown(
    rs: list[Record],
    key: Callable[[Record], Iterable[str]],
    metric: Callable[[list[Record]], tuple[int, int]],
) -> dict[str, Any]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in rs:
        for k in key(r):
            groups[k].append(r)
    return {k: by_rep(v, metric) for k, v in sorted(groups.items())}


def _slices(r: Record) -> list[str]:
    out = [f"level:{r['level']}"] + [f"category:{c}" for c in r["categories"]]
    out.append("unseen:yes" if r["unseen"] else "unseen:no")
    return out


# -- workflow level ------------------------------------------------------------------


def workflow_metrics(rs: list[Record]) -> dict[str, Any]:
    completed = [r for r in rs if r["completed"]]
    calls = sum(r["calls"] for r in rs)
    return {
        "workflows": len(rs),
        "task_completion_rate": by_rep(rs, _count(lambda r: r["completed"])),
        "status_counts": _counts(r["status"] for r in rs),
        "steps_to_completion": spread([r["steps"] for r in completed]),
        "steps_histogram": _counts(str(r["steps"]) for r in completed),
        "excess_steps": spread(
            [r["excess_steps"] for r in completed if r["excess_steps"] is not None]
        ),
        "unnecessary_call_rate": {
            **ratio(sum(r["unnecessary_calls"] for r in rs), calls),
            "definition": "executed calls not on any shortest remaining plan / all calls",
        },
        "forbidden_outcome_rate": by_rep(rs, _count(lambda r: r["forbidden_reached"])),
        "retry_rate": ratio(sum(r["router_retries"] for r in rs), calls),
        "replan_rate": ratio(sum(r["replans"] for r in rs), max(calls, 0)),
        "ask_human_rate": by_rep(rs, _count(lambda r: r["asked_human"])),
        "routing_failure_rate": ratio(
            sum(r["routing_failures"] for r in rs),
            sum(r["routing_failures"] + r["routing_decisions"] for r in rs),
        ),
        "fallback_rate": ratio(
            sum(r["fallback_decisions"] for r in rs), sum(r["routing_decisions"] for r in rs)
        ),
        "routing_latency_ms_per_decision": spread(
            [x for r in rs for x in r["routing_latencies_ms"]]
        ),
        "execution_latency_ms_per_call": spread(
            [x for r in rs for x in r["execution_latencies_ms"]]
        ),
        "total_latency_ms_per_workflow": spread([r["total_latency_ms"] for r in rs]),
        "usage": usage(rs),
    }


# -- robustness ----------------------------------------------------------------------


def _decision_key(r: Record) -> str:
    return r["canonical_capability_id"] or f"<{r['intent'] or r['error_type']}>"


def order_sensitivity(rs: list[Record]) -> dict[str, Any]:
    """Share of (case, repetition) groups whose decision changes across capability orders."""
    groups: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for r in rs:
        groups[(r["case_id"], r["repetition"])].append(r)
    changed = sum(1 for g in groups.values() if len({_decision_key(r) for r in g}) > 1)
    acc_changed = sum(1 for g in groups.values() if len({r["acceptable"] for r in g}) > 1)
    return {
        "decision_changed": ratio(changed, len(groups)),
        "correctness_changed": ratio(acc_changed, len(groups)),
        "accuracy_by_permutation": breakdown(
            rs, lambda r: [r["perturbation"]], _count(lambda r: r["acceptable"])
        ),
    }


def name_sensitivity(main: list[Record], names: list[Record]) -> dict[str, Any]:
    ref = {(r["case_id"], r["repetition"]): r for r in main}
    pairs = [
        (ref[(r["case_id"], r["repetition"])], r)
        for r in names
        if (r["case_id"], r["repetition"]) in ref
    ]
    changed = sum(1 for a, b in pairs if _decision_key(a) != _decision_key(b))
    a_acc = sum(a["acceptable"] for a, _ in pairs)
    b_acc = sum(b["acceptable"] for _, b in pairs)
    return {
        "pairs": len(pairs),
        "decision_changed": ratio(changed, len(pairs)),
        "accuracy_original": ratio(a_acc, len(pairs)),
        "accuracy_opaque_names": ratio(b_acc, len(pairs)),
        "accuracy_delta": (b_acc - a_acc) / len(pairs) if pairs else None,
    }


def distractor_curve(decisions: list[Record], workflows: list[Record]) -> dict[str, Any]:
    def level(r: Record) -> int:
        return int(r["perturbation"].split(":")[1])

    out: dict[str, Any] = {}
    for n in sorted({level(r) for r in decisions + workflows}):
        d = [r for r in decisions if level(r) == n]
        w = [r for r in workflows if level(r) == n]
        out[str(n)] = {
            "decisions": len(d),
            "routing_accuracy": by_rep(d, _count(lambda r: r["acceptable"])) if d else None,
            "valid_decision_rate": by_rep(d, _count(lambda r: r["valid"])) if d else None,
            "unnecessary_selection_rate": (
                by_rep(d, _count(lambda r: r["unnecessary"])) if d else None
            ),
            "routing_latency_ms": spread([r["routing_latency_ms"] for r in d]),
            "workflows": len(w),
            "task_completion_rate": by_rep(w, _count(lambda r: r["completed"])) if w else None,
            "unnecessary_call_rate": (
                ratio(sum(r["unnecessary_calls"] for r in w), sum(r["calls"] for r in w))
                if w
                else None
            ),
        }
    return out


def summarize(decisions: list[Record], workflows: list[Record]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    routers = sorted({r["router"] for r in decisions + workflows})
    for router in routers:
        d = [r for r in decisions if r["router"] == router]
        w = [r for r in workflows if r["router"] == router]
        main_d = [r for r in d if r["experiment"] == "main"]
        main_w = [r for r in w if r["experiment"] in ("main", "fallback")]
        entry: dict[str, Any] = {}
        if main_d:
            entry["decision"] = decision_metrics(main_d)
            entry["decision_by_slice"] = breakdown(
                main_d, _slices, _count(lambda r: r["acceptable"])
            )
        if main_w:
            entry["workflow"] = workflow_metrics(main_w)
            entry["workflow_by_slice"] = breakdown(
                main_w, _slices, _count(lambda r: r["completed"])
            )
            entry["failure_recovery_success"] = by_rep(
                main_w,
                _count(
                    lambda r: r["completed"], within=lambda r: "failure_recovery" in r["categories"]
                ),
            )
            entry["unseen_composition_success"] = by_rep(
                main_w, _count(lambda r: r["completed"], within=lambda r: r["unseen"])
            )
        order = [r for r in d if r["experiment"] == "order"]
        if order:
            entry["order_sensitivity"] = order_sensitivity(order)
        names = [r for r in d if r["experiment"] == "names"]
        if names:
            entry["name_sensitivity"] = name_sensitivity(main_d, names)
        dd = [r for r in d if r["experiment"] == "distractors"]
        dw = [r for r in w if r["experiment"] == "distractors"]
        if dd or dw:
            entry["distractor_robustness"] = distractor_curve(dd, dw)
        out[router] = entry
    return out


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
