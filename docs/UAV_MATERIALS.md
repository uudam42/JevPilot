# UAV Materials: Methodology

This document explains how the UAV materials workflow turns a natural-language
request into an engineering report, and what each step may and may not claim.
Code: `domains/uav_materials/` (domain) and `apps/uav_materials/` (application).

```text
request ─► interpret ─► search existing ─► decide ─┬─► report
                                                   └─► design (grid) ─► evaluate ─► report
```

## 1. Orchestration

The workflow is six ordinary JevPilot capabilities in the `uav_materials`
domain. They take no inputs (everything is read from the workflow state), and
their `is_applicable` preconditions make only the sensible next steps
available:

| Capability | Available when | Produces |
|---|---|---|
| `uavm.interpret_requirements` | request not yet interpreted | `uav.interpretation` |
| `uavm.search_existing_materials` | requirements interpreted | `uav.existing_search` |
| `uavm.assess_existing_candidates` | search done, no decision | `uav.decision` |
| `uavm.design_composite_candidate` | decision is *design* | `uav.design_search` |
| `uavm.evaluate_designed_candidate` | a designed candidate exists | `uav.design_assessment` |
| `uavm.generate_material_report` | every step the decision needs is done | `uav.report`, `uav.report_markdown` |

The router sees only a compact progress record (`WorkflowProgress`) and short
step summaries; large results stay in artifacts. The domain evaluator declares
success when the report exists. In **OFFLINE_DEMO** mode a scripted reference
policy (first applicable step in the order above) is served through the real
`JevRouter` pipeline by `FakeJevAdapter`. In **LIVE** mode Jev
(`TypeSafeJevAdapter`) chooses among the same capabilities. If the loop ends
early, the application still builds a report from the state and marks it
*incomplete* with the reason.

## 2. Requirement interpretation

An interpreter returns an `InterpretationPayload`: requirements, concerns,
application, `design_allowed`, missing information. The schema has **no field
for a material or a material property**. Every item is validated by
`parse_interpretation` before use:

- `source_text` must be a verbatim quote from the request;
- every number (limit, exposure time) must be written in that quote;
- the unit must be written in that quote (40 GPa cannot become 40 MPa);
- a limit is a hard requirement; a qualitative wish ("lightweight") is a soft
  preference with no number;
- the requirement must pass the domain validation (unit dimension, required test
  conditions such as the corrosion environment).

Failing items are dropped and listed as integrity findings. Soft preferences
get equal weights by a stated convention; interpreters never assign weights.
Two interpreters exist:

- `RuleBasedInterpreter`: deterministic phrase patterns (limits, directions,
  qualitative wishes, marine exposure); used offline;
- `LLMRequirementInterpreter` (application layer): Claude with the domain's
  prompt (`interpretation_system_prompt()`), same validation.

## 3. Requirement semantics

Each requirement is an `EngineeringRequirement`: property family, *aspect*
(tensile ultimate strength, normal stiffness, ductility, ...), *direction*,
limit, unit, hard/soft, conditions. The `RequirementResolver` maps it, per
material system, to the one property that may answer it:

| System | Resolves | Never |
|---|---|---|
| `isotropic_bulk` (all 46 records) | every aspect to its bulk property, any direction (isotropy assumption stated) | – |
| `continuous_fiber_ud_ply` | directional aspects only with material axes 1/2/12 | E11 → directionless modulus; S11T → directionless strength; failure strain → ductility |
| `continuous_fiber_laminate` | laminate axes x/y/xy → Ex/Ey/Gxy/νxy | quasi-isotropy inferred from a layup; ply S11T as laminate strength |

Outcomes are RESOLVED, AMBIGUOUS (underspecified for this system) or
UNSUPPORTED (no observable). A resolved requirement without a value is
MISSING. These three are never collapsed.

## 4. Existing materials

46 records built offline from committed raw extracts: MIL-HDBK-5J design
allowables (strict table parser), NRL AD0609618 tropical corrosion statements
and MIL-HDBK-5J / NASA temperature statements (quote-verified). Values keep
their original unit, statistical basis, test conditions and source pointer;
identity links (exact / alloy level) are recorded. Selection picks, per
requirement, the condition-compatible measurement with the best identity and
statistical basis and the most conservative value.

## 5. Evaluation

The unified evaluator (`evaluation.py`) applies the same logic to existing and
designed candidates: resolve → select → check hard constraints (satisfied /
violated / undetermined) → rank soft preferences (relative gap to the best
viable value, weighted RMS, pessimistic distance with missing preferences at
the worst observed penalty) → evidence assessment. The evidence layer
classifies a candidate as *infeasible* (a violation backed by sufficient
evidence), *evidence gap* (a hard constraint cannot be judged), *adequate*
(only with an explicit acceptable region) or *unclassified*; *design gap* is
reserved and never assigned automatically. Predictions are reported as
predictions (`evidence_type = predicted`) and never counted as measurements.

## 6. Decision policy

`ExistingMaterialAcceptancePolicy` (a **demonstration policy**, not a UAV
design standard) checks, for each existing candidate in ranking order: all
hard constraints satisfied; hard constraints backed by sufficient evidence;
pessimistic preference distance ≤ 0.5; preference coverage ≥ 50 %. Outcomes:
`use_existing`, `design` (none passes, design allowed, and the design space
can address at least one hard requirement), `needs_information` (no hard
limit stated) or `no_suitable_existing`. Candidates that fail only for lack of
evidence are named as evidence gaps that might qualify if the evidence were
obtained. All thresholds are configuration.

## 7. Bounded inverse design

The design target is the subset of the user's requirements that resolves for
`continuous_fiber_laminate`; the rest (strength, failure, corrosion,
temperature, directionless stiffness) is excluded and reported. The search
enumerates AS/IMLS carbon/epoxy laminates: V_f ∈ {0.50, 0.55, 0.60, 0.65} ×
seven symmetric layups ([0]s, [90]s, [0/90]s, [±45]s, [0/45/-45/90]s,
[0/0/45/-45]s, [0/60/-60]s), 0.127 mm plies. Each design is predicted by the
models below and evaluated by the unified evaluator. Selection: feasible first,
then pessimistic distance, distance, coverage, layup order, lower V_f
(distances rounded so that mathematically equal designs tie exactly). No
optimiser and no language model generate designs.

## 8. Physics models

- **Micromechanics** (`micromechanics.py`, NASA TM-83320 / TP-3290): ply density,
  E11, E22, G12, ν12, S11T, failure strain; reproduces the sources' worked examples.
- **Classical Lamination Theory** (`clt.py`, NASA RP-1351): Q, Q̄(θ), A/B/D,
  symmetric-laminate Ex, Ey, Gxy, νxy; reproduces RP-1351's printed Q, Q̄(45°), A
  and B; B = 0 checked for every symmetric stack.
- **Constituents** (`constituents.py`): AS fibre and IMLS epoxy from the NASA
  data bank, each value accepted only when two NASA documents agree.

Every laminate prediction's provenance keeps the chain: CLT equation → ply Q̄
matrices → micromechanics predictions → constituent values → documents.

## 9. Report and integrity

`build_report` copies values from the state's artifacts (records, predictions,
evaluations); `render_markdown` formats that structured report and nothing
else. Tests check that every number in the Markdown occurs in the structured
report, and that the structured values equal the dataset, model and evaluator
values. The report states the run mode, the orchestration trace, the decision
policy, evidence types per requirement, limitations and sources. No language
model writes any part of it.

## 10. What is not claimed

Designed laminates are virtual designs predicted by analytical models checked
against their sources' examples, not experimentally validated materials. Their
strength, failure, corrosion, water absorption and temperature behaviour are
not predicted. The 46-record dataset has gaps; the acceptance policy is
illustrative.
