"""Validation against the REAL TypeSafe Jev API: preflight, UAV runs, sanitized summary.

Needs ``TYPESAFE_API_KEY`` in the environment (never printed or stored). Stages::

    python -m experiments.routing.real_jev preflight                      # auth + model discovery
    python -m experiments.routing.benchmark --mode live --routers rule,jev \\
        --sample tiny_live_v1 --experiments main                          # Stage A: smoke
    python -m experiments.routing.benchmark --mode live --routers rule,jev \\
        --sample jev_live_v1 --experiments main,order,distractors \\
        --order-permutations 2 --distractor-levels 16,48                  # Stage B: benchmark
    python -m experiments.routing.benchmark --mode live --routers jev \\
        --sample jev_live_v1 --experiments main --repetitions 3            # Stage C: repeats
    python -m experiments.routing.benchmark --mode live --routers rule,jev \\
        --split eval --kinds workflow --experiments main                  # workflows
    python -m experiments.routing.real_jev uav --out <dir>                # UAV end to end
    python -m experiments.routing.real_jev publish --run smoke=<dir> --run benchmark=<dir> \\
        --run repeats=<dir> --run workflows=<dir> [--run fake=<offline dir>] --uav <dir>
    python -m experiments.routing.real_jev fully-live --out <file>        # Claude + Jev

``fully-live`` (FULLY_LIVE) needs ``ANTHROPIC_API_KEY`` too: real Claude through
LangChain interprets, real Jev routes. It first checks the configured Claude model and
one interpretation (stopping if the structured output fails validation), then runs the
UAV branches and the CLI demo. Outcomes are recorded as findings, never retried.

Labels: ``REAL_JEV`` is the real service through ``TypeSafeJevAdapter``; ``RULE_ROUTER``
the deterministic rule baseline; ``FAKE_JEV`` the offline fake adapter (infrastructure
test only). The main benchmark never uses a fallback chain. The published summary
(``experiments/routing/results/published/``) contains metrics only: no API key, no
Authorization header, no environment, no request or state content.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.routing.generalization.metrics import summarize
from integrations.redaction import SECRET_ENV_VARS, contains_secret, redact

PUBLISHED = Path(__file__).parent / "results" / "published"
ARTIFACT_VERSION = "jevpilot.real-jev-validation/1"
LABELS = {"jev": "REAL_JEV", "rule": "RULE_ROUTER"}

# UAV end-to-end cases (the requests of tests/apps/test_uav_end_to_end.py). Expected
# outcomes are those of the offline reference run, fixed before any live run.
UAV_CASES: dict[str, dict[str, Any]] = {
    "A_existing_material": {
        "request": (
            "I need a structural material for a UAV spar. The density must not exceed 3000 "
            "kg/m^3. It needs a stiffness of at least 65 GPa in both the x and y directions "
            "and an in-plane shear stiffness of at least 25 GPa. Prefer low density and high "
            "stiffness in the x and y directions. If no existing material is suitable, design "
            "a composite candidate."
        ),
        "expected_decision": "use_existing",
        "design_must_run": False,
    },
    "B_inverse_design": {"request": None, "expected_decision": "design", "design_must_run": True},
    "C_needs_information": {
        "request": (
            "I need a lightweight UAV structural material with high stiffness, low density, and "
            "good load-carrying capability. Find the closest existing material. If no existing "
            "material is sufficiently suitable, design a composite candidate and explain the "
            "result."
        ),
        "expected_decision": "needs_information",
        "design_must_run": False,
    },
    "D_unsupported_property": {
        "request": (
            "The maximum service temperature must be at least 80 °C. If no existing material is "
            "suitable, design a composite candidate."
        ),
        "expected_decision": "no_suitable_existing",
        "design_must_run": False,
    },
    "E_langchain_inverse_design": {
        "request": None,
        "expected_decision": "design",
        "design_must_run": True,
        "llm_backend": "langchain",
    },
}
DESIGN_STEPS = ("uavm.design_composite_candidate", "uavm.evaluate_designed_candidate")


def _key_state() -> str:
    import os

    return "configured" if os.environ.get("TYPESAFE_API_KEY", "").strip() else "missing"


# -- preflight ---------------------------------------------------------------------------------


def preflight() -> dict[str, Any]:
    from integrations.typesafe_jev import TypeSafeJevAdapter

    out: dict[str, Any] = {"TYPESAFE_API_KEY": _key_state()}
    try:
        checked = TypeSafeJevAdapter().verify()
    except Exception as exc:
        failure = getattr(exc, "details", {}).get("failure")
        out.update(
            authentication="failed" if failure == "authentication" else "not verified",
            error=f"{type(exc).__name__}: {redact(exc)[:300]}",
        )
        return out
    out.update(checked)
    return out


# -- UAV end to end ------------------------------------------------------------------------


def run_uav(out_dir: Path, *, max_steps: int = 30) -> dict[str, Any]:
    from apps.uav_materials import DEMO_REQUEST, RunMode, run_uav_material_workflow

    results = {}
    for name, case in UAV_CASES.items():
        request = case["request"] or DEMO_REQUEST
        backend = case.get("llm_backend", "native")
        reference = run_uav_material_workflow(request, llm_backend=backend, max_steps=max_steps)
        live = run_uav_material_workflow(
            request, mode=RunMode.LIVE_ROUTING, llm_backend=backend, max_steps=max_steps
        )
        log = live.routing_log()
        sequence = [r["capability"] or f"<{r['intent']}>" for r in log]
        ref_sequence = [r["capability"] or f"<{r['intent']}>" for r in reference.routing_log()]
        invoked = [c for c in sequence if not c.startswith("<")]
        results[name] = {
            "mode": live.mode.value,
            "llm_backend": backend,
            "interpreter": live.report.interpretation.interpreter
            if live.report.interpretation
            else None,
            "expected_decision": case["expected_decision"],
            "decision": live.decision,
            "decision_matches": live.decision == case["expected_decision"],
            "offline_reference_decision": reference.decision,
            "report_status": live.report.status,
            "workflow_status": str(live.state.status),
            "complete": live.complete,
            "design_ran": any(c in invoked for c in DESIGN_STEPS),
            "design_expected": case["design_must_run"],
            "capability_sequence": sequence,
            "reference_sequence": ref_sequence,
            "sequence_matches_reference": sequence == ref_sequence,
            "repeated_capability": len(invoked) != len(set(invoked)),
            "routing_decisions": len(log),
            "routing_errors": [r["status"] for r in log if r["status"].startswith("error")],
            "fallbacks": sum(r["status"] == "fallback" for r in log),
            "retries": sum(r["retries"] for r in log),
            "models": sorted({r["model"] for r in log if r["model"]}),
            "latency_ms": [r["latency_ms"] for r in log],
            "confidence": [r["confidence"] for r in log],
            "stop_reason": live.run.control.reason,
            "routing_log": log,
        }
        report_numbers_equal = reference.report.model_dump(
            exclude={"run", "orchestration"}
        ) == live.report.model_dump(exclude={"run", "orchestration"})
        results[name]["scientific_content_equals_offline"] = report_numbers_equal
        print(
            f"[{live.mode.value}] {name}: decision={live.decision} "
            f"(expected {case['expected_decision']}), report={live.report.status}, "
            f"steps={len(log)}",
            file=sys.stderr,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(UTC).isoformat(), "cases": results}
    _write_json(out_dir / "uav_real_jev.json", payload)
    return payload


# -- publish ---------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"value": round(num / den, 4) if den else None, "n": num, "of": den}


def _conf(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(statistics.fmean(xs), 4),
            "median": round(statistics.median(xs), 4),
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
        }

    valid = [r for r in records if r["valid"] and r["confidence"] is not None]
    return {
        "all_valid": stats([r["confidence"] for r in valid]),
        "when_acceptable": stats([r["confidence"] for r in valid if r["acceptable"]]),
        "when_not_acceptable": stats([r["confidence"] for r in valid if not r["acceptable"]]),
        "note": "self-reported by Jev; descriptive only, no calibration claim",
    }


def _api(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = [e for r in records for e in r.get("transport_events", [])]
    ok_calls = sum(r.get("model_calls") or 0 for r in records)
    kinds = Counter(e["kind"] for e in events)
    requests = ok_calls + len(events)
    return {
        "api_requests": requests,
        "failed_api_requests": len(events),
        "api_error_rate": _rate(len(events), requests),
        "rate_limited": _rate(kinds.get("rate_limited", 0), requests),
        "server_errors": kinds.get("server_error", 0),
        "timeouts": kinds.get("timeout", 0),
        "errors_by_kind": dict(sorted(kinds.items())),
        "retries": sum(r.get("transport_retries", 0) for r in records),
    }


def _decision_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    main = [r for r in records if r["experiment"] == "main"]
    s = summarize(records, [])
    router = records[0]["router"] if records else None
    m = s.get(router, {}) if router else {}
    d = m.get("decision", {})

    def v(block: Any) -> Any:
        return {k: block.get(k) for k in ("value", "numerator", "denominator")} if block else None

    by_slice = m.get("decision_by_slice", {})
    errors = Counter(r["error_type"] for r in main if r["error_type"])
    return {
        "decisions": len(main),
        "cases": len({r["case_id"] for r in main}),
        "repetitions": len({r["repetition"] for r in main}),
        "valid_typed_decision_rate": v(d.get("valid_decision_rate")),
        "acceptable_next_action_accuracy": v(d.get("routing_accuracy")),
        "preferred_action_rate": v(d.get("preferred_rate")),
        "forbidden_action_rate": v(d.get("forbidden_selection_rate")),
        "unnecessary_selection_rate": v(d.get("unnecessary_selection_rate")),
        "ask_human": {k: v(x) for k, x in d.get("ask_human", {}).items()},
        "finish": {k: v(x) for k, x in d.get("finish", {}).items()},
        "invalid_capability_rate": v(d.get("invalid_capability_rate")),
        "invalid_input_rate": v(d.get("invalid_input_rate")),
        "malformed_rate": v(d.get("malformed_rate")),
        "adapter_failure_rate": v(d.get("adapter_failure_rate")),
        "routing_errors_by_type": dict(sorted(errors.items())),
        "timeouts": errors.get("RouterTimeoutError", 0),
        "fallback_count": sum(bool(r["fallback_used"]) for r in main),
        "accuracy_by_level": {
            k: v(x) for k, x in by_slice.items() if k.startswith("L") or k.startswith("level")
        },
        "accuracy_by_slice": {k: v(x) for k, x in by_slice.items()},
        "latency_ms": {
            "p50": _pct([r["routing_latency_ms"] for r in main], 0.5),
            "p95": _pct([r["routing_latency_ms"] for r in main], 0.95),
            "max": _pct([r["routing_latency_ms"] for r in main], 1.0),
        },
        "confidence": _conf(main),
        "api": _api(records),
        "models": sorted({x for r in main for x in r.get("actual_models", [])}),
        "order_sensitivity": _strip(m.get("order_sensitivity")),
        "distractor_robustness": _strip(m.get("distractor_robustness")),
    }


def _strip(block: Any) -> Any:
    """Keep only value/numerator/denominator of nested ratio blocks (drops per-rep noise)."""
    if isinstance(block, dict):
        if "value" in block and "numerator" in block:
            return {k: block[k] for k in ("value", "numerator", "denominator")}
        return {k: _strip(x) for k, x in block.items() if k not in ("routing_latency_ms",)}
    return block


def _repeat_consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["experiment"] == "main":
            groups[r["case_id"]].append(r)
    reps = {len(g) for g in groups.values()}

    def key(r: dict[str, Any]) -> str:
        return str(r["canonical_capability_id"] or f"<{r['intent'] or r['error_type']}>")

    same = sum(1 for g in groups.values() if len({key(r) for r in g}) == 1)
    same_ok = sum(1 for g in groups.values() if len({r["acceptable"] for r in g}) == 1)
    per_rep = defaultdict(list)
    for r in records:
        if r["experiment"] == "main":
            per_rep[r["repetition"]].append(r["acceptable"])
    return {
        "cases": len(groups),
        "repetitions_per_case": sorted(reps),
        "identical_decision_across_repetitions": _rate(same, len(groups)),
        "identical_correctness_across_repetitions": _rate(same_ok, len(groups)),
        "accuracy_per_repetition": [
            round(sum(x) / len(x), 4) for _, x in sorted(per_rep.items()) if x
        ],
    }


def _workflow_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    s = summarize([], records)
    w = s[records[0]["router"]]["workflow"]
    return {
        "workflows": w["workflows"],
        "task_completion_rate": _strip(w["task_completion_rate"]),
        "status_counts": w["status_counts"],
        "unnecessary_call_rate": _strip(w["unnecessary_call_rate"]),
        "routing_failure_rate": _strip(w["routing_failure_rate"]),
        "fallback_rate": _strip(w["fallback_rate"]),
        "ask_human_rate": _strip(w["ask_human_rate"]),
        "forbidden_outcome_rate": _strip(w["forbidden_outcome_rate"]),
        "steps_to_completion": {k: w["steps_to_completion"].get(k) for k in ("n", "mean", "max")},
        "excess_steps": {k: w["excess_steps"].get(k) for k in ("n", "mean", "max")},
        "latency_ms_per_decision": {
            "p50": _pct([x for r in records for x in r.get("routing_latencies_ms", [])], 0.5),
            "p95": _pct([x for r in records for x in r.get("routing_latencies_ms", [])], 0.95),
        },
        "api": _api(records),
    }


def publish(runs: dict[str, Path], uav: Path | None, out: Path) -> dict[str, Any]:
    manifests = {k: json.loads((d / "manifest.json").read_text()) for k, d in runs.items()}
    stages: dict[str, Any] = {}
    real: dict[str, Any] = {}
    for stage, directory in runs.items():
        man = manifests[stage]
        decisions = _read_jsonl(directory / "decisions.jsonl")
        workflows = _read_jsonl(directory / "workflows.jsonl")
        block: dict[str, Any] = {
            "run_mode": man["mode"],
            "benchmark_version": man["benchmark_version"],
            "split": man["split"],
            "split_digest": man["split_digest"],
            "sample": man.get("sample"),
            "experiments": man["experiments"],
            "repetitions": man["repetitions"],
            "timestamp": man["timestamp"],
            "git_commit": (man.get("git") or {}).get("commit"),
            "routers": {},
        }
        for name in sorted({r["router"] for r in decisions + workflows}):
            label = "FAKE_JEV" if man["mode"] == "offline" and name == "jev" else LABELS[name]
            d = [r for r in decisions if r["router"] == name]
            w = [r for r in workflows if r["router"] == name]
            entry: dict[str, Any] = {}
            if d:
                entry["decision"] = _decision_block(d)
                if man["repetitions"] > 1:
                    entry["repeat_consistency"] = _repeat_consistency(d)
            if w:
                entry["workflow"] = _workflow_block(w)
            block["routers"][label] = entry
            if label == "REAL_JEV":
                slot = man["routers"].get(name, {})
                real = {
                    "preflight": {
                        k: slot.get("preflight", {}).get(k)
                        for k in (
                            "requested_model",
                            "model_source",
                            "listed_models",
                            "release_date",
                            "sdk_version",
                            "authentication",
                        )
                    },
                    "adapter": {
                        k: (slot.get("config", {}).get("adapter") or {}).get(k)
                        for k in ("adapter_version", "request_format", "mode", "retries")
                    },
                }
        stages[stage] = block
    uav_block = None
    if uav is not None:
        raw = json.loads((uav / "uav_real_jev.json").read_text())
        uav_block = {
            name: {k: x for k, x in case.items() if k != "routing_log"}
            | {"routing_log": case["routing_log"]}
            for name, case in raw["cases"].items()
        }
    artifact = {
        "artifact": ARTIFACT_VERSION,
        "label": "REAL_JEV",
        "date": datetime.now(UTC).date().isoformat(),
        "provider": "typesafe",
        "jev": real,
        "stages": stages,
        "uav_end_to_end": uav_block,
        "notes": [
            "REAL_JEV: real TypeSafe System One API through TypeSafeJevAdapter; no fallback in "
            "the main benchmark.",
            "RULE_ROUTER: deterministic baseline; FAKE_JEV: offline fake adapter (infrastructure "
            "test, not model performance).",
            "Cases were frozen before the live runs (benchmarks/routing/samples); nothing was "
            "tuned on live results.",
            "Confidence values are Jev's self-reports; no calibration is claimed.",
            "Contains metrics only: no API key, Authorization header, environment variable "
            "value, request or state content.",
        ],
    }
    text = json.dumps(artifact, indent=2, sort_keys=False) + "\n"
    _refuse_secrets(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return artifact


def _refuse_secrets(text: str) -> None:
    if contains_secret(text) or redact(text) != text:
        raise SystemExit("refusing to write: the artifact contains credential-like text")
    for name in SECRET_ENV_VARS:
        if f'"{name}"' in text:
            raise SystemExit(f"refusing to write: the artifact mentions {name}")


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, default=str) + "\n"
    _refuse_secrets(text)
    path.write_text(text, encoding="utf-8")


# -- fully live: real Claude (LangChain) + real Jev ------------------------------------------

FULLY_LIVE = "FULLY_LIVE"
FULLY_LIVE_BRANCHES = (
    "A_existing_material",
    "B_inverse_design",
    "C_needs_information",
    "D_unsupported_property",
)
_VOLATILE = {"run", "orchestration", "status", "status_note"}


def _requirements(outcome: Any) -> list[dict[str, Any]]:
    rows = []
    for r in outcome.requirements:
        meta = r.provenance.sources[0].metadata if r.provenance else {}
        rows.append(
            {
                "label": r.label,
                "priority": r.priority.value,
                "value": r.value,
                "unit": r.unit,
                "quote": meta.get("quote"),
            }
        )
    return rows


def _numbers_quoted(outcome: Any, request: str) -> bool:
    """Independent re-check: each limit is written in its quote, and the quote in the request."""
    from domains.uav_materials.interpretation import numbers_in

    norm = " ".join(request.split()).casefold()
    for r in outcome.requirements:
        quote = (r.provenance.sources[0].metadata.get("quote") if r.provenance else "") or ""
        if " ".join(quote.split()).casefold() not in norm:
            return False
        for v in (r.value, r.upper):
            if v is not None and not any(
                abs(n - v) <= 1e-9 * max(1.0, abs(v)) for n in numbers_in(quote)
            ):
                return False
    return True


def _report_numbers(report: Any) -> list[float]:
    """Every number in the structured report (same traversal as the end-to-end tests)."""
    import re

    pattern = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
    out: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, int | float):
            out.append(float(x))
        elif isinstance(x, str):
            out.extend(float(t) for t in pattern.findall(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)

    walk(report.model_dump(mode="json"))
    return out


def _untraced(report: Any, markdown: str) -> list[str]:
    """Numbers in the Markdown that the structured report does not contain."""
    from domains.uav_materials.reporting import numbers_in_markdown

    known = _report_numbers(report)
    missing = []
    for token in numbers_in_markdown(markdown):
        pct = token.endswith("%")
        text = token.rstrip("%")
        value = float(text) / (100 if pct else 1)
        decimals = len(text.split(".")[1]) if "." in text else 0
        tol = 0.5 * 10**-decimals / (100 if pct else 1) + 1e-12
        if not any(abs(n - value) <= tol for n in known):
            missing.append(token)
    return missing


class _Replay:
    """Returns a recorded interpretation (to re-run the science deterministically offline)."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    def describe(self) -> dict[str, Any]:
        return dict(self.outcome.interpreter)

    def interpret(self, request_text: str) -> Any:
        return self.outcome


def claude_smoke() -> dict[str, Any]:
    """Configured Claude model exists; one real interpretation through LangChain validates."""
    import time

    from apps.uav_materials import DEMO_REQUEST
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter
    from integrations.anthropic_llm import AnthropicLLMAdapter
    from integrations.langchain_chat import chat_model

    checked = AnthropicLLMAdapter().verify()
    interpreter = LangChainRequirementInterpreter(chat_model())
    t0 = time.perf_counter()
    outcome = interpreter.interpret(DEMO_REQUEST)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "label": f"{FULLY_LIVE} (interpretation only)",
        "claude_preflight": {
            k: checked.get(k) for k in ("requested_model", "model_id", "display_name")
        },
        "interpreter": {k: outcome.interpreter.get(k) for k in ("name", "backend", "kind")}
        | {
            "chat_model": outcome.interpreter.get("chat_model"),
            "served_model": outcome.interpreter.get("served_model"),
        },
        "status": outcome.status,
        "schema_valid": outcome.status != "failed",
        "error": outcome.error,
        "requirements": _requirements(outcome),
        "integrity_findings": [f.model_dump() for f in outcome.findings],
        "missing_information": list(outcome.missing_information),
        "all_numbers_quoted_from_request": _numbers_quoted(outcome, DEMO_REQUEST),
        "latency_ms": latency_ms,
    }


def _fully_live_record(name: str, request: str, live: Any, expected: str | None) -> dict[str, Any]:
    from apps.uav_materials import RunMode, execute_workflow
    from apps.uav_materials.workflow import offline_components
    from domains.uav_materials.workflow import INTERPRETATION
    from jevpilot import TraceEventType

    artifact = live.state.latest_artifact(INTERPRETATION)
    outcome = artifact.content if artifact is not None else None
    log = live.routing_log()
    sequence = [r["capability"] or f"<{r['intent']}>" for r in log]
    invoked = [c for c in sequence if not c.startswith("<")]
    offered_ok = all(
        e.payload["decision"].get("capability_id")
        in (e.payload.get("available_capabilities") or [])
        for e in live.run.trace
        if e.type is TraceEventType.ROUTING_DECISION and e.payload["decision"].get("capability_id")
    )
    interp_ms = next(
        (
            round(o.execution_time * 1000, 1)
            for o in live.state.observations
            if o.capability_id == "uavm.interpret_requirements"
        ),
        None,
    )
    replay_equal = None
    if outcome is not None:
        _, router = offline_components()
        replay = execute_workflow(
            request, interpreter=_Replay(outcome), router=router, mode=RunMode.OFFLINE_DEMO
        )
        replay_equal = live.report.model_dump(exclude=_VOLATILE) == replay.report.model_dump(
            exclude=_VOLATILE
        )
    run = live.report.run
    return {
        "label": FULLY_LIVE,
        "app_mode": live.mode.value,
        "llm_backend": "langchain",
        "request_case": name,
        "claude": {
            "requested_model": ((run.interpreter.get("chat_model") or {}).get("model")),
            "served_model": outcome.interpreter.get("served_model") if outcome else None,
            "chat_model_class": (run.interpreter.get("chat_model") or {}).get("class"),
            "interpretation_latency_ms": interp_ms,
        },
        "jev": {
            "router": sorted({r["router"] for r in log if r["router"]}),
            "requested_model": (run.router.get("adapter") or {}).get("model"),
            "served_models": sorted({r["model"] for r in log if r["model"]}),
        },
        "interpretation": {
            "status": outcome.status if outcome else None,
            "schema_valid": outcome is not None and outcome.status != "failed",
            "requirements": _requirements(outcome) if outcome else [],
            "integrity_findings": [f.model_dump() for f in outcome.findings] if outcome else [],
            "missing_information": list(outcome.missing_information) if outcome else [],
            "design_allowed": outcome.design_allowed if outcome else None,
            "all_numbers_quoted_from_request": (
                _numbers_quoted(outcome, request) if outcome else None
            ),
        },
        "expected_decision_offline_reference": expected,
        "decision": live.decision,
        "decision_matches_offline_reference": (
            None if expected is None else live.decision == expected
        ),
        "report_status": live.report.status,
        "workflow_status": str(live.state.status),
        "stop_reason": live.run.control.reason,
        "capability_sequence": sequence,
        "preconditions_respected": offered_ok
        and not any("InvalidCapabilityError" in r["status"] for r in log),
        "design_ran": any(c in invoked for c in DESIGN_STEPS),
        "design_ran_only_after_design_decision": (
            not any(c in invoked for c in DESIGN_STEPS) or live.decision == "design"
        ),
        "repeated_capability": len(invoked) != len(set(invoked)),
        "routing_errors": [r["status"] for r in log if r["status"].startswith("error")],
        "fallbacks": sum(r["status"] == "fallback" for r in log),
        "untraced_report_numbers": _untraced(live.report, live.markdown),
        "science_equals_offline_replay_of_same_interpretation": replay_equal,
        "report_provenance": {
            "mode": run.mode,
            "note": run.note,
            "interpreter_kind": run.interpreter.get("kind"),
            "interpreter_backend": run.interpreter.get("backend"),
            "router_provider": (run.router.get("adapter") or {}).get("provider"),
        },
        "routing_latency_ms": [r["latency_ms"] for r in log],
        "routing_confidence": [r["confidence"] for r in log],
        "routing_log": log,
    }


def run_fully_live(out: Path) -> dict[str, Any]:
    import os
    import subprocess
    import tempfile
    import time

    from apps.uav_materials import DEMO_REQUEST, RunMode, run_uav_material_workflow
    from domains.uav_materials.reporting import MaterialWorkflowReport

    smoke = claude_smoke()
    print(
        f"[{FULLY_LIVE}] Claude smoke: status={smoke['status']} "
        f"served={smoke['interpreter']['served_model']} "
        f"requirements={len(smoke['requirements'])} latency_ms={smoke['latency_ms']}",
        file=sys.stderr,
    )
    runs: dict[str, Any] = {}
    if smoke["schema_valid"]:
        for name in FULLY_LIVE_BRANCHES:
            case = UAV_CASES[name]
            request = case["request"] or DEMO_REQUEST
            t0 = time.perf_counter()
            try:
                live = run_uav_material_workflow(
                    request, mode=RunMode.LIVE, llm_backend="langchain"
                )
            except Exception as exc:  # a finding, not a retry
                runs[name] = {
                    "label": FULLY_LIVE,
                    "error": f"{type(exc).__name__}: {redact(exc)[:300]}",
                }
                continue
            record = _fully_live_record(name, request, live, case["expected_decision"])
            record["wall_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            runs[name] = record
            print(
                f"[{FULLY_LIVE}] {name}: decision={live.decision} "
                f"(offline reference {case['expected_decision']}), report={live.report.status}",
                file=sys.stderr,
            )
        with tempfile.TemporaryDirectory() as tmp:  # the normal CLI demo command
            cli = Path(sys.executable).parent / "jevpilot"
            files = {
                k: Path(tmp) / f
                for k, f in (("json", "r.json"), ("md", "r.md"), ("log", "l.jsonl"))
            }
            t0 = time.perf_counter()
            done = subprocess.run(
                [
                    str(cli),
                    "uav-materials",
                    "--live",
                    "--llm-backend",
                    "langchain",
                    "--quiet",
                    "--json",
                    str(files["json"]),
                    "--output",
                    str(files["md"]),
                    "--routing-log",
                    str(files["log"]),
                ],  # fmt: skip
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            wall = round((time.perf_counter() - t0) * 1000, 1)
            record = {
                "label": FULLY_LIVE,
                "command": "jevpilot uav-materials --live --llm-backend langchain (demo request)",
                "exit_code": done.returncode,
                "stderr_status": redact(done.stderr.strip())[-300:],
                "wall_time_ms": wall,
            }
            if files["json"].exists():
                report = MaterialWorkflowReport.model_validate_json(files["json"].read_text())
                log = [json.loads(x) for x in files["log"].read_text().splitlines() if x.strip()]
                record |= {
                    "decision": report.decision.outcome if report.decision else None,
                    "report_status": report.status,
                    "report_mode": report.run.mode,
                    "interpreter_backend": report.run.interpreter.get("backend"),
                    "claude_requested_model": (report.run.interpreter.get("chat_model") or {}).get(
                        "model"
                    ),
                    "jev_served_models": sorted({r["model"] for r in log if r["model"]}),
                    "capability_sequence": [r["capability"] or f"<{r['intent']}>" for r in log],
                    "untraced_report_numbers": _untraced(report, files["md"].read_text()),
                    "routing_latency_ms": [r["latency_ms"] for r in log],
                    "routing_log": log,
                }
            runs["E_cli_demo"] = record
            print(f"[{FULLY_LIVE}] E_cli_demo: exit={done.returncode}", file=sys.stderr)
    payload = {
        "artifact": "jevpilot.fully-live-validation/1",
        "label": FULLY_LIVE,
        "date": datetime.now(UTC).date().isoformat(),
        "path": "real Claude via LangChain -> real TypeSafe Jev -> JevPilot controller -> "
        "UAV capabilities -> scientific models -> report",
        "claude_smoke": smoke,
        "runs": runs,
        "notes": [
            "FULLY_LIVE runs only: no scripted chat model, RuleRouter or FakeJevAdapter result "
            "is included here.",
            "Expected decisions are the offline reference outcomes fixed before any live run; "
            "differences are findings, nothing was tuned or retried.",
            "Contains no API key, Authorization header or environment variable value.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, payload)
    return payload


# -- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    u = sub.add_parser("uav")
    u.add_argument("--out", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--run", action="append", default=[], help="stage=results_dir")
    pub.add_argument("--uav", type=Path)
    pub.add_argument("--out", type=Path)
    fl = sub.add_parser("fully-live")
    fl.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.command == "preflight":
        result = preflight()
        for key, value in result.items():
            print(f"{key}: {value}")
        return 0 if result.get("authentication") == "ok" else 1
    if _key_state() == "missing" and args.command == "uav":
        print("TYPESAFE_API_KEY: missing", file=sys.stderr)
        return 2
    if args.command == "uav":
        run_uav(args.out)
        return 0
    if args.command == "fully-live":
        from apps.uav_materials.workflow import live_requirements

        missing = live_requirements("langchain")
        if missing:
            print("FULLY_LIVE needs: " + "; ".join(missing), file=sys.stderr)
            return 2
        target = args.out or PUBLISHED / f"fully_live_{datetime.now(UTC):%Y-%m-%d}.json"
        result = run_fully_live(target)
        print(f"written: {target}")
        return 0 if result["claude_smoke"]["schema_valid"] else 1
    runs = {k: Path(v) for k, v in (item.split("=", 1) for item in args.run)}
    out = args.out or PUBLISHED / f"real_jev_{datetime.now(UTC):%Y-%m-%d}.json"
    publish(runs, args.uav, out)
    print(f"published: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
