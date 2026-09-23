"""UAV materials domain (iteration 1: schema and target property representation).

Depends on JevPilot's public API; JevPilot never imports this package.
"""

from domains.uav_materials.module import MaterialsReducer, UAVMaterialsDomain
from domains.uav_materials.profile import (
    Importance,
    OperatingEnvironment,
    Operator,
    Priority,
    PropertyRequirement,
    SearchFeatureVector,
    TargetMaterialProfile,
)
from domains.uav_materials.schema import (
    CandidateOrigin,
    MaterialCandidate,
    MaterialFamily,
    MaterialRecord,
    Measurement,
    MeasurementBasis,
    MissingReason,
    ProvenanceStatus,
    Quantity,
    TestConditions,
)
from domains.uav_materials.search import MaterialSearchResult, search_materials
from domains.uav_materials.state import UAVMaterialsState

__all__ = [
    "CandidateOrigin",
    "Importance",
    "MaterialCandidate",
    "MaterialFamily",
    "MaterialRecord",
    "MaterialSearchResult",
    "MaterialsReducer",
    "Measurement",
    "MeasurementBasis",
    "MissingReason",
    "OperatingEnvironment",
    "Operator",
    "Priority",
    "PropertyRequirement",
    "ProvenanceStatus",
    "Quantity",
    "SearchFeatureVector",
    "TargetMaterialProfile",
    "TestConditions",
    "UAVMaterialsDomain",
    "UAVMaterialsState",
    "search_materials",
]
