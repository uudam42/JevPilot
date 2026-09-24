"""Minimal capabilities proving the domain runs on the unmodified JevPilot loop.

Both take *structured* input; there is no natural-language extraction, no
retrieval and no LLM. Their outputs enter state through the domain reducer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from domains.uav_materials.profile import TargetMaterialProfile
from domains.uav_materials.schema import MaterialCandidate
from domains.uav_materials.state import UAVMaterialsState
from jevpilot import (
    Artifact,
    Capability,
    CapabilityResult,
    CapabilitySpec,
    ExecutionContext,
    SourceRef,
    WorkflowState,
)

DOMAIN = "uav_materials"


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: TargetMaterialProfile


class CandidatesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[MaterialCandidate]
    source: str  # where the records came from, e.g. "synthetic test fixtures"


class ConstructTargetProfile(Capability):
    spec = CapabilitySpec(
        id="uavm.construct_target_profile",
        name="construct_target_profile",
        description="Validate structured material requirements (hard constraints, soft "
        "preferences, ranges, weights, operating environment) and set them as the target "
        "material profile.",
        domain=DOMAIN,
        input_schema=ProfileInput,
        output_schema=TargetMaterialProfile,
        effects=("sets target_profile",),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        # structured set-up only; request-driven workflows use the workflow capabilities
        return isinstance(state, UAVMaterialsState) and state.request_text is None

    def execute(self, inputs: ProfileInput, ctx: ExecutionContext) -> CapabilityResult:
        profile = inputs.profile
        return CapabilityResult(
            output=profile,
            artifacts=(
                Artifact(
                    name=f"target_profile:{profile.profile_id}",
                    kind="target_material_profile",
                    content=profile.model_dump(mode="json"),
                ),
            ),
        )


class RegisterCandidateMaterials(Capability):
    spec = CapabilitySpec(
        id="uavm.register_candidate_materials",
        name="register_candidate_materials",
        description="Add already-structured material candidate records to the workflow.",
        domain=DOMAIN,
        input_schema=CandidatesInput,
        effects=("extends candidate_materials",),
    )

    def is_applicable(self, state: WorkflowState) -> bool:
        # structured set-up only; request-driven workflows use the workflow capabilities
        return isinstance(state, UAVMaterialsState) and state.request_text is None

    def execute(self, inputs: CandidatesInput, ctx: ExecutionContext) -> CapabilityResult:
        return CapabilityResult(
            output=tuple(inputs.candidates),
            sources=(SourceRef(kind="dataset", identifier=inputs.source),),
        )


def all_capabilities() -> list[Capability]:
    return [ConstructTargetProfile(), RegisterCandidateMaterials()]
