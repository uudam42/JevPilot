"""Domain state: generic WorkflowState plus the UAV-materials objects built so far.

Two uses share this state type:

* structured set-up (``target_profile``, ``candidate_materials``), iteration 1;
* the request-driven workflow (``request_text``, ``progress``): natural-language
  request → requirements → existing-material search → decision → optional
  composite design → report. Large results live in artifacts; the router sees
  only the compact :class:`WorkflowProgress`.
"""

from __future__ import annotations

from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate
from jevpilot import FrozenModel, WorkflowState


class WorkflowProgress(FrozenModel):
    """Router-visible progress of the request-driven workflow."""

    requirements: str = "pending"  # pending | ok | partial | empty | failed
    existing_search: str = "pending"  # pending | done
    decision: str = "pending"  # pending | use_existing | design | needs_information | ...
    design_search: str = "pending"  # pending | designed | no_feasible_design | ...
    design_evaluation: str = "pending"  # pending | done
    report: str = "pending"  # pending | done
    summary: str = ""  # one line about the latest step


class UAVMaterialsState(WorkflowState):
    target_profile: TargetMaterialProfile | None = None
    candidate_materials: tuple[MaterialCandidate, ...] = ()
    request_text: str | None = None  # set: request-driven workflow
    progress: WorkflowProgress = WorkflowProgress()
