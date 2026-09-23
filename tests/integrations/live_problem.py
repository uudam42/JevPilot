"""The minimal routing problem used by the live smoke tests (iteration spec §3/§4).

Goal: summary statistics. State: dataset loaded and cleaned, statistics not
computed. Three offered capabilities from the benchmark catalog.
The only acceptable action is ``tk.compute_statistics``.
"""

from __future__ import annotations

from experiments.routing.generalization.world import initial_state, load_catalog
from jevpilot import CapabilitySpec, WorkflowState

OFFERED = ("tk.compute_statistics", "tk.generate_report", "tk.quick_summary")
ACCEPTABLE = {"tk.compute_statistics"}


def smoke_problem() -> tuple[WorkflowState, list[CapabilitySpec]]:
    catalog = load_catalog()
    state = initial_state(
        "Compute descriptive statistics for the loaded dataset.",
        {"source": "store://projects/alpha/measurements"},
        ["dataset_loaded", "records_cleaned"],
        workflow_id="wf_live_smoke",
    )
    return state.evolve(status="running"), [catalog.get(c).spec() for c in OFFERED]
