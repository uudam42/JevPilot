"""UAV materials application: natural-language request → engineering report.

from apps.uav_materials import run_uav_material_workflow
result = run_uav_material_workflow()          # OFFLINE_DEMO, the built-in request
print(result.markdown)
"""

from apps.uav_materials.workflow import (
    DEMO_REQUEST,
    LiveModeUnavailable,
    RunMode,
    UAVWorkflowResult,
    execute_workflow,
    run_uav_material_workflow,
)

__all__ = [
    "DEMO_REQUEST",
    "LiveModeUnavailable",
    "RunMode",
    "UAVWorkflowResult",
    "execute_workflow",
    "run_uav_material_workflow",
]
