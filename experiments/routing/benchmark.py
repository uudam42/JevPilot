"""Routing generalization benchmark: same benchmark, state and capabilities; different router.

    # offline: RuleRouter + FAKE model adapters (infrastructure test, no network, free)
    python -m experiments.routing.benchmark --mode offline --routers rule,llm,jev --split eval

    # live: real Claude and real Jev, strict single-model mode, 5 repetitions
    python -m experiments.routing.benchmark --mode live --routers rule,llm,jev \\
        --split eval --repetitions 5 --strict

    # robustness and reliability experiments
    python -m experiments.routing.benchmark --experiments main,order,names,distractors
    python -m experiments.routing.benchmark --experiments fallback --systems 'jev>rule,llm>rule'

See docs/BENCHMARK_DESIGN.md and docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jevpilot
from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.metrics import summarize
from experiments.routing.generalization.routers import (
    MODES,
    ROUTERS,
    LiveOptions,
    RouterSlot,
    build_chain,
    build_slot,
)
from experiments.routing.generalization.runner import EXPERIMENTS, RunConfig, run
from experiments.routing.generalization.world import BENCHMARK_DIR, load_catalog
from jevpilot import stable_digest
from jevpilot.routing import ROUTING_PROMPT_VERSION

BENCHMARK_VERSION = "jevpilot.routing.generalization/1"
DEFAULT_OUT = Path(__file__).parent / "results"
OFFLINE_FAULTS = {"malformed": 0.03, "invalid_capability": 0.03, "invalid_inputs": 0.03}


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--mode", choices=MODES, default="offline")
    p.add_argument(
        "--routers", type=_csv, default=list(ROUTERS), help="comma-separated: rule,llm,jev"
    )
    p.add_argument("--split", choices=ds.SPLITS, default="eval")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument(
        "--experiments",
        type=_csv,
        default=["main"],
        help=f"comma-separated subset of {','.join(EXPERIMENTS)}",
    )
    p.add_argument("--kinds", type=_csv, default=["decision", "workflow"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample",
        help="named decision-case sample, e.g. tiny_live_v1 "
        "(decision kind only; its split overrides --split)",
    )
    p.add_argument("--order-permutations", type=int, default=3)
    p.add_argument(
        "--distractor-levels", type=lambda v: [int(x) for x in _csv(v)], default=[8, 16, 32, 48]
    )
    p.add_argument(
        "--systems",
        type=_csv,
        default=["jev>rule", "llm>rule", "jev>llm>rule"],
        help="fallback chains for the 'fallback' experiment",
    )
    strict = p.add_mutually_exclusive_group()
    strict.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="single-model attribution (default)",
    )
    strict.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="allow provider-side model substitution (production mode)",
    )
    p.add_argument(
        "--faults", action="store_true", help="offline only: inject faults into the fake adapters"
    )
    p.add_argument("--llm-model")
    p.add_argument("--llm-effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--llm-max-tokens", type=int, default=16000)
    p.add_argument("--llm-price", help="USD per MTok 'input,output' (records cost)")
    p.add_argument("--jev-model")
    p.add_argument("--jev-price-input", type=float, help="USD per MTok of input")
    p.add_argument("--pricing-source", help="where/when the prices were taken from")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-traces", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    bad = set(args.experiments) - set(EXPERIMENTS)
    if bad:
        p.error(f"unknown experiments {sorted(bad)}")
    if args.mode == "live" and args.faults:
        p.error("--faults injects fake behaviour and is only allowed in offline mode")
    args.sample_data = None
    if args.sample:
        args.sample_data = ds.load_sample(args.sample)
        args.split = args.sample_data["split"]
        args.kinds = ["decision"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = load_catalog()
    started = datetime.now(UTC)
    run_id = (
        f"{started:%Y%m%dT%H%M%SZ}-{args.mode}-{args.split}-"
        + stable_digest([args.routers, args.experiments, args.seed, args.repetitions])[:6]
    )
    out = args.out / run_id
    live = LiveOptions(
        strict=args.strict,
        llm_model=args.llm_model,
        llm_effort=args.llm_effort,
        llm_max_tokens=args.llm_max_tokens,
        llm_price=tuple(float(x) for x in _csv(args.llm_price)) if args.llm_price else None,  # type: ignore[arg-type]
        jev_model=args.jev_model,
        jev_price_input=args.jev_price_input,
        pricing_source=args.pricing_source,
        timeout_s=args.timeout,
    )
    faults = OFFLINE_FAULTS if args.faults else {}
    slots = {n: build_slot(n, args.mode, faults=faults, live=live) for n in args.routers}
    systems = (
        [build_chain(c, _components(c, slots, args, faults, live)) for c in args.systems]
        if "fallback" in args.experiments
        else []
    )
    for s in [*slots.values(), *systems]:
        if not s.available:
            print(f"[unavailable] {s.name}: {s.unavailable}", file=sys.stderr)
    cfg = RunConfig(
        split=args.split,
        experiments=args.experiments,
        repetitions=args.repetitions,
        seed=args.seed,
        order_permutations=args.order_permutations,
        distractor_levels=args.distractor_levels,
        kinds=args.kinds,
        trace_dir=None if args.no_traces else out / "traces",
        decision_case_ids=(
            frozenset(args.sample_data["decision_cases"]) if args.sample_data else None
        ),
        progress=None if args.quiet else (lambda m: print(f"  … {m}", file=sys.stderr)),
    )
    results = run(list(slots.values()), catalog, cfg, systems)
    summary = summarize(results.decisions, results.workflows)
    manifest = build_manifest(run_id, started, args, slots, systems, results)
    write(out, manifest, results, summary)
    print(format_table(summary, manifest))
    print(f"\nresults: {out}")
    return 0


def _components(
    chain: str,
    slots: dict[str, RouterSlot],
    args: argparse.Namespace,
    faults: dict[str, float],
    live: LiveOptions,
) -> dict[str, RouterSlot]:
    have = dict(slots)
    for part in chain.split(">"):
        have.setdefault(part, build_slot(part, args.mode, faults=faults, live=live))
    return have


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=BENCHMARK_DIR,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=BENCHMARK_DIR,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _version(module: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:
        return None


def build_manifest(
    run_id: str,
    started: datetime,
    args: argparse.Namespace,
    slots: dict[str, RouterSlot],
    systems: list[RouterSlot],
    results: Any,
) -> dict[str, Any]:
    catalog_files = ("catalog.json", "distractors.json")
    specs = ds.load_split(args.split)
    return {
        "run_id": run_id,
        "timestamp": started.isoformat(),
        "git": git_state(),
        "jevpilot_version": jevpilot.__version__,
        "benchmark_version": BENCHMARK_VERSION,
        "mode": args.mode,
        "evaluation_type": "live model routing"
        if args.mode == "live"
        else "OFFLINE INFRASTRUCTURE TEST: fake adapters, not model performance",
        "strict_single_model": args.strict,
        "split": args.split,
        "split_digest": ds.split_digest(args.split),
        "catalog_digests": {
            f: stable_digest(json.loads((BENCHMARK_DIR / f).read_text())) for f in catalog_files
        },
        "cases": {
            "workflow": [s.case_id for s in specs],
            "decision": sorted({r["case_id"] for r in results.decisions}),
            "unseen_composition": [s.case_id for s in specs if s.unseen_composition],
        },
        "sample": (
            {"name": args.sample, "digest": stable_digest(args.sample_data)}
            if args.sample_data
            else None
        ),
        "experiments": args.experiments,
        "kinds": args.kinds,
        "repetitions": args.repetitions,
        "seeds": {
            "benchmark_seed": args.seed,
            "item_seed": "sha256([seed, experiment, perturbation, case_id, repetition])[:8]",
            "order_seed": "sha256(['order', seed, case_id, k])",
            "name_seed": "sha256(['names', seed, workflow_case_id])",
            "distractor_seed": "sha256(['scale', seed, case_id, total])",
            "fault_seed": "item_seed (offline fakes only)",
        },
        "order_permutations": args.order_permutations,
        "distractor_levels": args.distractor_levels,
        "offline_fault_rates": OFFLINE_FAULTS if args.faults else {},
        "prompt_version": ROUTING_PROMPT_VERSION,
        "routers": {name: _slot_record(s) for name, s in slots.items()},
        "fallback_systems": {s.name: _slot_record(s) for s in systems},
        "main_comparison_uses_fallback": False,
        "determinism": (
            "Offline runs are deterministic given the seed (except latencies). Live model outputs "
            "are not assumed deterministic; repetitions are recorded separately."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "anthropic_sdk": _version("anthropic"),
            "typesafe_sdk": _version("typesafe-sdk"),
            "pydantic": _version("pydantic"),
        },
    }


def _slot_record(s: RouterSlot) -> dict[str, Any]:
    return {
        "kind": s.kind,
        "status": "available" if s.available else "unavailable",
        "unavailable_reason": s.unavailable,
        "preflight": s.preflight,
        "config": s.config,
    }


def write(out: Path, manifest: dict[str, Any], results: Any, summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, rows in (("decisions", results.decisions), ("workflows", results.workflows)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for fp, req in sorted(results.requests.items()):
            fh.write(json.dumps({"fingerprint": fp, "request": req}, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = summary_rows(summary, manifest)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["router"])
        writer.writeheader()
        writer.writerows(rows)


def _v(block: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(block, dict) or key not in block:
            return None
        block = block[key]
    return block


def summary_rows(summary: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for router, m in summary.items():
        rows.append(
            {
                "run_id": manifest["run_id"],
                "mode": manifest["mode"],
                "split": manifest["split"],
                "router": router,
                "decisions": _v(m, "decision", "decisions"),
                "valid_decision_rate": _v(m, "decision", "valid_decision_rate", "value"),
                "routing_accuracy": _v(m, "decision", "routing_accuracy", "value"),
                "routing_accuracy_std": _v(
                    m, "decision", "routing_accuracy", "across_repetitions", "std"
                ),
                "preferred_rate": _v(m, "decision", "preferred_rate", "value"),
                "invalid_capability_rate": _v(m, "decision", "invalid_capability_rate", "value"),
                "ask_human_recall": _v(m, "decision", "ask_human", "recall", "value"),
                "ask_human_precision": _v(m, "decision", "ask_human", "precision", "value"),
                "finish_recall": _v(m, "decision", "finish", "recall", "value"),
                "workflows": _v(m, "workflow", "workflows"),
                "task_completion_rate": _v(m, "workflow", "task_completion_rate", "value"),
                "unseen_composition_success": _v(m, "unseen_composition_success", "value"),
                "failure_recovery_success": _v(m, "failure_recovery_success", "value"),
                "unnecessary_call_rate": _v(m, "workflow", "unnecessary_call_rate", "value"),
                "mean_steps": _v(m, "workflow", "steps_to_completion", "mean"),
                "order_decision_changed": _v(m, "order_sensitivity", "decision_changed", "value"),
                "name_accuracy_delta": _v(m, "name_sensitivity", "accuracy_delta"),
                "routing_latency_ms_mean": _v(m, "decision", "routing_latency_ms", "mean"),
                "input_tokens": _v(m, "decision", "usage", "input_tokens"),
                "output_tokens": _v(m, "decision", "usage", "output_tokens"),
            }
        )
    return rows


def format_table(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    def pct(x: Any) -> str:
        return "    -" if x is None else f"{100 * x:5.1f}%"

    def num(x: Any) -> str:
        return "    -" if x is None else f"{x:7.2f}"

    lines = [
        f"{manifest['evaluation_type']} · split={manifest['split']} · "
        f"repetitions={manifest['repetitions']}"
    ]
    for name, r in manifest["routers"].items():
        if r["status"] != "available":
            lines.append(f"  {name}: UNAVAILABLE ({r['unavailable_reason']})")
    header = (
        f"  {'router':<14} {'VDR':>6} {'RA':>6} {'pref':>6} {'ask-R':>6} {'fin-R':>6} "
        f"{'TCR':>6} {'unseen':>6} {'recov':>6} {'UCR':>6} {'steps':>7} {'ms/dec':>7}"
    )
    lines.append(header)
    for router, m in summary.items():
        lines.append(
            f"  {router:<14} {pct(_v(m, 'decision', 'valid_decision_rate', 'value')):>6} "
            f"{pct(_v(m, 'decision', 'routing_accuracy', 'value')):>6} "
            f"{pct(_v(m, 'decision', 'preferred_rate', 'value')):>6} "
            f"{pct(_v(m, 'decision', 'ask_human', 'recall', 'value')):>6} "
            f"{pct(_v(m, 'decision', 'finish', 'recall', 'value')):>6} "
            f"{pct(_v(m, 'workflow', 'task_completion_rate', 'value')):>6} "
            f"{pct(_v(m, 'unseen_composition_success', 'value')):>6} "
            f"{pct(_v(m, 'failure_recovery_success', 'value')):>6} "
            f"{pct(_v(m, 'workflow', 'unnecessary_call_rate', 'value')):>6} "
            f"{num(_v(m, 'workflow', 'steps_to_completion', 'mean')):>7} "
            f"{num(_v(m, 'decision', 'routing_latency_ms', 'mean')):>7}"
        )
    robust = [
        (r, m)
        for r, m in summary.items()
        if any(k in m for k in ("order_sensitivity", "name_sensitivity", "distractor_robustness"))
    ]
    for router, m in robust:
        parts = []
        if "order_sensitivity" in m:
            parts.append(
                "order-changed " + pct(m["order_sensitivity"]["decision_changed"]["value"])
            )
        if "name_sensitivity" in m:
            ns = m["name_sensitivity"]
            parts.append(
                "opaque-names RA "
                + pct(ns["accuracy_opaque_names"]["value"])
                + " vs "
                + pct(ns["accuracy_original"]["value"])
            )
        if "distractor_robustness" in m:
            curve = m["distractor_robustness"]
            parts.append(
                "distractors RA "
                + " ".join(
                    f"{n}:{pct(_v(c, 'routing_accuracy', 'value')).strip()}"
                    for n, c in curve.items()
                )
            )
        lines.append(f"  {router:<14} " + " · ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
