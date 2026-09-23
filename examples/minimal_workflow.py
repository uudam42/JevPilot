"""Run the arithmetic demo domain end to end and print the trace.

python examples/minimal_workflow.py
"""

from __future__ import annotations

from domains.demo import ArithmeticDomain, NumberState, routing_rules
from jevpilot import RuleRouter, Runtime


def main() -> None:
    runtime = Runtime()
    domain = runtime.load(ArithmeticDomain())
    assert isinstance(domain, ArithmeticDomain)

    controller = runtime.controller(RuleRouter(routing_rules()))
    result = controller.run(domain.new_workflow(start=3, target=20))

    for event in result.trace:
        if event.type == "state_snapshot":
            continue
        detail = ""
        if event.type == "routing_decision":
            d = event.payload["decision"]
            detail = f"{d['capability_id']} {d['inputs']}  ({d['reason']})"
        elif event.type == "observation":
            o = event.payload["observation"]
            detail = f"success={o['success']} result={o['result']}"
        elif event.type == "control_decision":
            detail = event.payload["control"]["action"]
        print(f"[{event.seq:02d}] step={event.step} {event.type:<20} {detail}")

    state = result.state
    assert isinstance(state, NumberState)
    print(f"\nstatus={state.status} value={state.value} steps={state.step}")
    print(f"provenance records: {len(state.provenance)}")


if __name__ == "__main__":
    main()
