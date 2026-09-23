"""Two unrelated domains, one unmodified core.

python examples/cross_domain.py
"""

from __future__ import annotations

import domains.demo as arithmetic
import domains.stats_demo as stats
from jevpilot import RuleRouter, Runtime


def main() -> None:
    runtime = Runtime()
    # Loaded by string path: the core never imports domain code statically.
    arith = runtime.load("domains.demo:ArithmeticDomain")
    stat = runtime.load("domains.stats_demo:StatsDomain")
    print("registered capabilities:", sorted(s.id for s in runtime.capabilities.list()))

    r1 = runtime.controller(RuleRouter(arithmetic.routing_rules()), domains=["arithmetic"]).run(
        arith.new_workflow(start=5, target=42)  # type: ignore[attr-defined]
    )
    print(f"arithmetic: {r1.state.status} after {r1.state.step} steps")

    r2 = runtime.controller(RuleRouter(stats.routing_rules()), domains=["stats"]).run(
        stat.new_workflow(true_mean=10.0, noise=3.0, ci_half_width=0.5)  # type: ignore[attr-defined]
    )
    print(f"stats:      {r2.state.status} after {r2.state.step} steps")
    report = r2.state.latest_artifact("report")
    if report and report.provenance:
        print(f"            {report.content}")
        print(f"            report provenance derived_from={report.provenance.derived_from}")


if __name__ == "__main__":
    main()
