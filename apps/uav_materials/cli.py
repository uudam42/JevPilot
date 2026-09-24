"""Command line for the UAV materials workflow.

jevpilot uav-materials --demo                      # offline, no credentials
jevpilot uav-materials --demo --request "..."      # your own request, offline
jevpilot uav-materials --live --request "..."      # Claude + Jev (needs API keys)
python -m apps.uav_materials --demo --output report.md --json result.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from apps.uav_materials.workflow import (
    DEMO_REQUEST,
    LiveModeUnavailable,
    RunMode,
    run_uav_material_workflow,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jevpilot uav-materials",
        description="Natural-language UAV material request -> existing-material search -> "
        "composite design if needed -> evidence-aware engineering report.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo",
        action="store_true",
        help="OFFLINE_DEMO mode (default): deterministic interpreter and scripted routing; "
        "no credentials or network",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="LIVE mode: Claude interprets the request and Jev routes (needs ANTHROPIC_API_KEY "
        "and TYPESAFE_API_KEY; never falls back to the demo)",
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument("--request", help="the request text (default: the built-in demo request)")
    source.add_argument("--request-file", type=Path, help="read the request from a file")
    p.add_argument("--output", type=Path, help="write the Markdown report to this file")
    p.add_argument("--json", type=Path, help="write the structured report (JSON) to this file")
    p.add_argument(
        "--quiet", action="store_true", help="do not print the report (use with --output)"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    request = (
        args.request_file.read_text(encoding="utf-8")
        if args.request_file
        else args.request or DEMO_REQUEST
    )
    mode = RunMode.LIVE if args.live else RunMode.OFFLINE_DEMO
    try:
        result = run_uav_material_workflow(request, mode=mode)
    except LiveModeUnavailable as exc:
        print(f"jevpilot uav-materials: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(result.markdown + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(result.to_json() + "\n", encoding="utf-8")
    if not args.quiet:
        print(result.markdown)
    status = f"[{result.mode.value}] decision: {result.decision}; report: {result.report.status}"
    print(status, file=sys.stderr)
    return 0 if result.report.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
