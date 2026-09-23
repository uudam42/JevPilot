# UAV materials domain (iteration 1: schema)

Represents *what properties a UAV design wants* (`TargetMaterialProfile`) and
*what properties a material has* (`MaterialRecord`), with physical units, test
conditions, explicit missing data and JevPilot provenance. Depends on
JevPilot's public API only; JevPilot never imports it.

| Module | Content |
|---|---|
| `units.py` | small explicit unit table (dimension + SI conversion; `pint` evaluated, deferred) |
| `properties.py` | property registry: name → category, dimension, direction, required test conditions |
| `schema.py` | `Measurement`, `TestConditions`, `MaterialRecord`, `MaterialCandidate` (existing / proposed / composite / virtual) |
| `profile.py` | `PropertyRequirement`, `TargetMaterialProfile`, `OperatingEnvironment`; `SearchFeatureVector` / `FeatureExtractor` (interface only) |
| `search.py` | `search_materials()` interface + minimal hard-constraint screen (no ranking yet) |
| `buoyancy.py` | density ratio and net buoyant force per volume (physics helpers, not a simulator) |
| `state.py`, `capabilities.py`, `module.py` | `UAVMaterialsState`, two structured-input capabilities, reducer, evaluator |
| `fixtures/` | 7 **fictional** materials for software tests, never engineering data |

Run: `python examples/uav_target_profile.py`. Tests: `tests/domains/uav_materials/`.
