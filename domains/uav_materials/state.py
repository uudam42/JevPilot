"""Domain state: generic WorkflowState plus the UAV-materials objects built so far.

Only what this iteration needs. Later iterations add evaluated and selected
candidates and simulation results as new fields, not by changing the core.
"""

from __future__ import annotations

from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate
from jevpilot import WorkflowState


class UAVMaterialsState(WorkflowState):
    target_profile: TargetMaterialProfile | None = None
    candidate_materials: tuple[MaterialCandidate, ...] = ()
