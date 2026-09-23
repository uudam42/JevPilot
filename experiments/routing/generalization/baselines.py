"""Baseline routing policies for the toolkit world.

``toolkit_rules`` is the RuleRouter baseline. It was written by reading **only
the dev split** (its goal wording and reference plans). It must not import the
dataset, the oracle or any split file, and ``tests/benchmark/test_leakage.py``
enforces this. It maps goal keywords to the facts the goal needs, then fires
the first pipeline rule whose facts are missing and whose prerequisites hold.
It is deterministic, inspectable and keyed on the canonical capability ids,
so it is name-dependent by construction.

``naive_request_policy`` drives the *offline fake* model adapters. It is a
generic token-overlap heuristic over the JSON routing request and exists to
exercise the harness offline. It is not a model of Jev or of any LLM, and its
scores are not evidence about them.
"""

from __future__ import annotations

import re
from typing import Any

from jevpilot import Rule, WorkflowState
from jevpilot.routing import RoutingRequest


def _facts(s: WorkflowState) -> set[str]:
    return set(s.context.get("facts", ()))


def _goal(s: WorkflowState) -> str:
    return s.goal.description.lower()


def _wants(s: WorkflowState) -> set[str]:
    """Target facts implied by the goal wording (keywords seen in dev goals)."""
    g, w = _goal(s), set()
    if "metadata" in g:
        w.add("metadata_fetched")
    if "declared schema" in g:
        w.add("schema_validated")
    if "load the dataset" in g:
        w.add("dataset_loaded")
    if "descriptive statistics" in g or "statistics" in g:
        w.add("statistics_computed")
    if "report" in g:
        w.add("report_generated")
    if "chart" in g:
        w.add("plot_created")
    if "report" in g and "chart" in g:
        w.add("report_has_figures")
    if "trend" in g:
        w.add("trend_fitted")
    if "anomalous" in g:
        w.add("outliers_flagged")
    if "export" in g:
        w.add("table_exported")
    if "compare" in g:
        w.add("groups_compared")
    if "reference table" in g:
        w.add("reference_merged")
    if "archive" in g:
        w.add("results_archived")
    if "owner" in g:
        w.add("owner_notified")
    if "internal team" in g:
        w.add("shared_internal")
    if "dashboard" in g:
        w.add("dashboard_rendered")
    return w


# (capability, fact it adds, facts it needs), later pipeline stages first.
_PIPELINE: list[tuple[str, str, tuple[str, ...]]] = [
    ("tk.share_internal", "shared_internal", ("report_generated",)),
    ("tk.archive_results", "results_archived", ("report_generated",)),
    (
        "tk.generate_illustrated_report",
        "report_has_figures",
        ("statistics_computed", "plot_created"),
    ),
    ("tk.generate_report", "report_generated", ("statistics_computed",)),
    ("tk.generate_plot", "plot_created", ("statistics_computed",)),
    ("tk.render_dashboard", "dashboard_rendered", ("statistics_computed", "trend_fitted")),
    ("tk.compare_groups", "groups_compared", ("statistics_computed",)),
    ("tk.fit_trend", "trend_fitted", ("records_cleaned", "units_normalized")),
    ("tk.detect_outliers", "outliers_flagged", ("records_cleaned",)),
    ("tk.export_table", "table_exported", ("records_cleaned",)),
    ("tk.merge_reference", "reference_merged", ("records_cleaned",)),
    ("tk.compute_statistics", "statistics_computed", ("records_cleaned",)),
    ("tk.drop_invalid_rows", "records_cleaned", ("schema_validated",)),
    ("tk.clean_records", "records_cleaned", ("dataset_loaded",)),
    ("tk.normalize_units", "units_normalized", ("dataset_loaded",)),
    ("tk.validate_schema", "schema_validated", ("dataset_loaded",)),
    ("tk.notify_owner", "owner_notified", ()),
    ("tk.fetch_metadata", "metadata_fetched", ()),
    ("tk.load_dataset", "dataset_loaded", ()),
]

# What each target needs upstream (from the dev reference plans).
_NEEDS = {
    "shared_internal": {"report_generated"},
    "results_archived": {"report_generated"},
    "report_has_figures": {"plot_created", "statistics_computed"},
    "report_generated": {"statistics_computed"},
    "plot_created": {"statistics_computed"},
    "dashboard_rendered": {"statistics_computed", "trend_fitted"},
    "groups_compared": {"statistics_computed"},
    "trend_fitted": {"records_cleaned", "units_normalized"},
    "outliers_flagged": {"records_cleaned"},
    "table_exported": {"records_cleaned"},
    "reference_merged": {"records_cleaned"},
    "statistics_computed": {"records_cleaned"},
    "records_cleaned": {"dataset_loaded"},
    "units_normalized": {"dataset_loaded"},
    "schema_validated": {"dataset_loaded"},
}


def _needed(s: WorkflowState) -> set[str]:
    needed, frontier = set(), list(_wants(s))
    while frontier:
        fact = frontier.pop()
        if fact not in needed:
            needed.add(fact)
            frontier.extend(_NEEDS.get(fact, ()))
    return needed


def _failed_permanently(s: WorkflowState, cid: str) -> bool:
    return any(
        o.capability_id == cid
        and not o.success
        and o.error is not None
        and "permanently" in o.error.message
        for o in s.observations
    )


def _inputs(cid: str, s: WorkflowState) -> dict[str, Any]:
    p = s.goal.parameters
    return {
        "tk.load_dataset": {"source": p.get("source")},
        "tk.load_dataset_mirror": {"source": p.get("source")},
        "tk.fetch_metadata": {"source": p.get("source")},
        "tk.merge_reference": {"reference": p.get("reference")},
        "tk.compare_groups": {"group_field": p.get("group_field")},
        "tk.generate_plot": {"kind": p.get("plot_kind", "bar")},
        "tk.export_table": {"format": p.get("export_format", "csv")},
        "tk.notify_owner": {"channel": p.get("channel", "email")},
    }.get(cid, {})


_PARAM_NEEDED = {
    "tk.load_dataset": "source",
    "tk.load_dataset_mirror": "source",
    "tk.fetch_metadata": "source",
    "tk.merge_reference": "reference",
    "tk.compare_groups": "group_field",
}


# Rules that must not fire when another target covers them (seen in dev_l3_illustrated).
_BLOCKED_BY = {"tk.generate_report": "report_has_figures"}


def _rule(cid: str, adds: str, needs: tuple[str, ...]) -> Rule:
    param = _PARAM_NEEDED.get(cid)
    blocker = _BLOCKED_BY.get(cid)

    def when(s: WorkflowState) -> bool:
        f = _facts(s)
        needed = _needed(s)
        return (
            adds in needed
            and adds not in f
            and set(needs) <= f
            and (blocker is None or blocker not in needed)
            and (param is None or param in s.goal.parameters)
            and not _failed_permanently(s, cid)
        )

    return Rule(cid, when=when, inputs=lambda s: _inputs(cid, s), reason=f"goal needs {adds}")


def toolkit_rules() -> list[Rule]:
    rules = [_rule(cid, adds, needs) for cid, adds, needs in _PIPELINE]
    # dev_l4_permanent_load: fall back to the mirror once the primary store is gone.
    rules.append(
        Rule(
            "tk.load_dataset_mirror",
            when=lambda s: (
                "dataset_loaded" in _needed(s)
                and "dataset_loaded" not in _facts(s)
                and "source" in s.goal.parameters
                and _failed_permanently(s, "tk.load_dataset")
            ),
            inputs=lambda s: _inputs("tk.load_dataset_mirror", s),
            reason="primary store permanently unavailable; use the mirror",
        )
    )
    return rules


# -- offline fake policy (infrastructure only) -------------------------------------

_TOKEN = re.compile(r"[a-z]+")
_FACT = re.compile(r"fact '([^']+)'")


def naive_request_policy(request: RoutingRequest) -> dict[str, Any]:
    """Pick the applicable capability whose text overlaps most with the goal."""
    facts = set(request.state_summary.context.get("facts", ()))
    goal = set(_TOKEN.findall(request.goal["description"].lower()))
    failed = {a.capability_id for a in request.previous_actions if a.success is False}
    best, best_score = None, 0
    for cap in request.available_capabilities:
        needs = {m for p in cap.preconditions for m in _FACT.findall(p)}
        adds = {m for e in cap.effects for m in _FACT.findall(e)}
        if not needs <= facts or not adds - facts or cap.id in failed:
            continue
        text = set(_TOKEN.findall(f"{cap.name} {cap.description} {' '.join(adds)}".lower()))
        score = len(goal & text)
        if score > best_score:
            best, best_score = cap, score
    if best is None:
        return {
            "action": "finish",
            "capability_id": None,
            "inputs": {},
            "reason": "no match",
            "confidence": 0.3,
        }
    params = request.goal["parameters"]
    inputs: dict[str, Any] = {}
    for field, prop in ((best.input_schema or {}).get("properties") or {}).items():
        values = [v for v in params.values() if isinstance(v, str)]
        enum = prop.get("enum")
        if enum:
            inputs[field] = next((v for v in values if v in enum), enum[0])
        elif values:
            inputs[field] = params.get(field, values[0])
    return {
        "action": "invoke",
        "capability_id": best.id,
        "inputs": inputs,
        "reason": f"token overlap {best_score}",
        "confidence": 0.5,
    }
