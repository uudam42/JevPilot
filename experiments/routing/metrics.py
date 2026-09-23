"""Aggregate metrics over benchmark records.

Every rate is reported with its numerator and denominator so that small
samples are visible. Unknown quantities (e.g. token usage from routers that
report none) stay ``None``. They are never counted as zero.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

Record = dict[str, Any]


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "p95": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": p95,
    }


USAGE_FIELDS = ("model_calls", "input_tokens", "output_tokens", "estimated_cost_usd")


def usage_totals(records: Sequence[Record]) -> dict[str, Any]:
    """Sum model usage; a field is ``None`` unless at least one record reported it."""
    totals: dict[str, Any] = {}
    for key in USAGE_FIELDS:
        known = [r[key] for r in records if r.get(key) is not None]
        totals[key] = sum(known) if known else None
    return totals


def decision_metrics(records: list[Record]) -> dict[str, Any]:
    """Single-step routing quality: R(S, C) → D against expected behaviour."""
    n = len(records)
    valid = [r for r in records if r["valid"]]
    attempts = sum(r["attempts"] for r in records)
    attempt_errors = Counter(e for r in records for e in r["attempt_errors"])
    confidence_pairs = [
        (r["confidence"], r["acceptable"]) for r in valid if r["confidence"] is not None
    ]
    return {
        "decisions": n,
        # final decisions (after any configured fallback)
        "valid_decision_rate": rate(len(valid), n),
        "routing_accuracy": rate(sum(r["acceptable"] for r in records), n),
        "routing_accuracy_given_valid": rate(sum(r["acceptable"] for r in valid), len(valid)),
        "forbidden_selection_rate": rate(sum(r["forbidden"] for r in records), n),
        "unnecessary_selection_rate": rate(sum(r["unnecessary"] for r in records), n),
        "no_action_rate": rate(sum(r["valid"] and r["capability_id"] is None for r in records), n),
        "fallback_rate": rate(sum(r["fallback_used"] for r in records), n),
        # every router attempt, including failed primaries behind a fallback
        "attempts": attempts,
        "invalid_capability_rate": rate(attempt_errors["InvalidCapabilityError"], attempts),
        "invalid_input_rate": rate(attempt_errors["InvalidRoutingInputError"], attempts),
        "malformed_rate": rate(attempt_errors["MalformedRoutingDecisionError"], attempts),
        "timeout_rate": rate(attempt_errors["RouterTimeoutError"], attempts),
        "adapter_error_rate": rate(attempt_errors["RouterAdapterError"], attempts),
        "attempt_errors": dict(sorted(attempt_errors.items())),
        "routing_latency_ms": distribution([r["routing_latency_ms"] for r in records]),
        "confidence": {
            # stored for later calibration analysis; not treated as correctness
            "reported": len(confidence_pairs),
            "mean_when_acceptable": _mean([c for c, ok in confidence_pairs if ok]),
            "mean_when_not_acceptable": _mean([c for c, ok in confidence_pairs if not ok]),
        },
        "usage": usage_totals(records),
    }


def workflow_metrics(records: list[Record]) -> dict[str, Any]:
    """Whole-workflow behaviour: completion, efficiency, latency split, control flow."""
    n = len(records)
    decisions = sum(r["routing_decisions"] for r in records)
    steps = sum(r["steps"] for r in records)
    completed = [r for r in records if r["completed"]]
    return {
        "workflows": n,
        "task_completion_rate": rate(len(completed), n),
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "steps_to_completion": distribution([r["steps"] for r in completed]),
        "steps_histogram": dict(sorted(Counter(r["steps"] for r in completed).items())),
        "excess_steps": distribution(
            [r["excess_steps"] for r in completed if r["excess_steps"] is not None]
        ),
        "unnecessary_calls": {
            "total": sum(r["unnecessary_calls"] for r in records),
            "per_workflow": distribution([r["unnecessary_calls"] for r in records]),
        },
        "routing_failure_rate": rate(
            sum(r["routing_failures"] for r in records),
            decisions + sum(r["routing_failures"] for r in records),
        ),
        "fallback_rate": rate(sum(r["fallback_decisions"] for r in records), decisions),
        "retry_rate": rate(sum(r["retries"] for r in records), steps),
        "replan_rate": rate(sum(r["replans"] for r in records), steps),
        "routing_latency_ms_per_decision": distribution(
            [x for r in records for x in r["routing_latencies_ms"]]
        ),
        "execution_latency_ms_per_call": distribution(
            [x for r in records for x in r["execution_latencies_ms"]]
        ),
        "routing_latency_ms_per_workflow": distribution([r["routing_latency_ms"] for r in records]),
        "execution_latency_ms_per_workflow": distribution(
            [r["execution_latency_ms"] for r in records]
        ),
        "total_latency_ms_per_workflow": distribution([r["total_latency_ms"] for r in records]),
        "usage": usage_totals(records),
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None
