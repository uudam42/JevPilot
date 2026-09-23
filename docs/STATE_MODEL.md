# State Model

All core data types are immutable Pydantic v2 models (`frozen=True`,
`extra="forbid"`) defined under `jevpilot/core/`. Ordered collections are
tuples. Updates always produce new objects via `state.evolve(...)` /
`model_copy(update=...)`, and in the loop they go only through `StateManager`.

```text
WorkflowState ─┬─ goal: Goal
               ├─ context, requirements, constraints
               ├─ observations: (Observation…) ──┬─ provenance: Provenance ── sources: (SourceRef…)
               │                                 ├─ artifacts: (Artifact…) ── provenance
               │                                 ├─ effects: StateEffects ── candidates, context/uncertainty updates
               │                                 └─ error: ErrorInfo | None, uncertainty: Uncertainty | None
               ├─ artifacts, candidate_solutions      (accumulated from observations)
               ├─ evaluations: (EvaluationResult…)
               ├─ uncertainty: {subject: Uncertainty}
               ├─ provenance: (Provenance…)           (deduplicated ledger)
               ├─ plan: Plan | None
               ├─ history: (HistoryEntry(step, RoutingDecision, observation_id, ControlDecision, routing_error)…)
               └─ step, status, metadata
```

## WorkflowState (`core/state.py`)

| Field | Meaning |
|---|---|
| `workflow_id` | Unique id, shared by every observation and provenance record of the run |
| `goal` | `Goal(description, parameters, success_criteria, metadata)`. Only the domain interprets `parameters` and `success_criteria` |
| `context` | Free-form working memory, merged from `StateEffects.context_updates` |
| `requirements`, `constraints` | Structured `Requirement` / `Constraint` (`spec` dict, `priority` / `hard`) |
| `observations` | Every `Observation`, successful or not, in order |
| `artifacts` | Every artifact produced, each with provenance |
| `candidate_solutions` | Proposed answers (`Candidate`: content, score, uncertainty, provenance) |
| `evaluations` | Every `EvaluationResult`, in order |
| `uncertainty` | Latest `Uncertainty` per named subject |
| `provenance` | Ledger of all provenance records that entered the state |
| `plan` | Latest `Plan` (with `revision`) |
| `history` | One `HistoryEntry` per loop iteration: decision, observation id, control decision |
| `step` | Number of executed capabilities |
| `status` | `pending → running → succeeded / failed / awaiting_human / cancelled` |

Helpers: `evolve()`, `last_observation`, `last_evaluation`,
`artifacts_of_kind()`, `latest_artifact()`.

**Extension.** Domains subclass `WorkflowState` to add typed fields. The
subclass survives every core transition. Domain `StateReducer`s update those
fields.

## Transition function

`StateManager.update(S_t, decision, O_t) → S_{t+1}`:

1. Reject observations from another workflow, and artifacts or candidates without provenance.
2. Append the observation, its artifacts and candidates.
3. Merge `context_updates` and `uncertainty_updates`.
4. Add new provenance records to the ledger, deduplicated by id.
5. Append a `HistoryEntry` and increment `step`.
6. Apply domain reducers in registration order.

Other transitions: `start`, `record_no_action`, `record_routing_failure`, `apply_evaluation`,
`record_control`, `set_plan`, `set_status`. All are pure.

## Observation (`core/observation.py`)

The result of executing one routing decision, successful or not.

| Field | Meaning |
|---|---|
| `capability_id`, `decision_id`, `workflow_id`, `step` | What was run, and why |
| `success`, `result`, `error: ErrorInfo(type, message, retryable, details)` | Outcome |
| `uncertainty` (`confidence` property) | Reliability of `result` |
| `artifacts` | Produced artifacts (provenance stamped) |
| `effects: StateEffects` | Proposed `context_updates`, `candidates`, `uncertainty_updates` |
| `provenance` | Execution provenance (always present) |
| `execution_time`, `timestamp`, `metadata` | Execution metadata |

## Artifact (`core/artifact.py`)

Any non-trivial output, stored inline (`content`) or by reference (`uri`).
`kind` follows `ArtifactKind` conventions (json, text, table, dataset, plot,
simulation_result, file, model, report, object), but any string is accepted.
Other fields: `media_type`, `schema_ref`, `uncertainty`, `provenance` and
`metadata`.

## Provenance (`core/provenance.py`)

| Field | Meaning |
|---|---|
| `workflow_id`, `step`, `execution_id` | When, within the workflow |
| `capability_id`, `capability_version`, `domain` | What produced it |
| `inputs`, `inputs_digest` | Validated inputs and their SHA-256 digest |
| `sources: (SourceRef(kind, identifier, version, uri)…)` | External origins such as a tool, model, database, dataset, human or publication |
| `derived_from` | Upstream provenance ids, which form the lineage graph |
| `started_at`, `finished_at`, `created_at` | Timing |
| `metadata` | Executor id, router id, and so on |

Lineage example from `stats_demo`: the report's provenance lists the summary's
provenance in `derived_from`. The summary's provenance lists the dataset's, and
the dataset's provenance carries `SourceRef(kind="generator")`.

## Uncertainty (`core/uncertainty.py`)

A generic, optional-everything record: `kind` (model, measurement,
data_quality, missing_information, conflict, unspecified), `confidence ∈ [0,1]`,
`interval`, `dispersion`, `missing`, `conflicts`, `notes` and `metadata`. The
producer defines what `interval` and `dispersion` mean (for example "95% CI",
"standard error") and records it in `metadata`. The core never interprets them.

## EvaluationResult (`core/evaluation.py`)

| Field | Meaning |
|---|---|
| `evaluator_id`, `step` | Who evaluated, and when |
| `goal_progress ∈ [0,1]` | Estimated progress toward the goal |
| `constraint_status: (ConstraintStatus(constraint_id, satisfied: bool/None, detail)…)` | Constraint checks |
| `quality`, `confidence` | Domain-defined quality score, and confidence in this assessment |
| `remaining_gaps` | What is still missing |
| `recommendation: ControlAction` | continue, retry, replan, terminate_success, terminate_failure or human_intervention |
| `rationale`, `metadata` | Explanation |

## Decisions (`core/decision.py`, `core/control.py`, `core/plan.py`)

- `RoutingDecision(capability_id | None, inputs, intent, reason, confidence, alternatives, router_id, metadata)`.
  `None` means "no action", and `intent` says why: `finish`, `ask_human` or
  `idle` (`invoke` when a capability is named). A retry copies the original
  decision and sets `metadata.retry_of`.
- `RoutingOutcome(decision, attempts)` is returned by `Router.route()`. Each
  `RoutingAttempt` records the router, success or error, latency, model,
  `ModelUsage` and, for model routers, the serialised `RoutingRequest`.
  Outcomes go into the trace, not into state.
- `ControlDecision(action: ControlAction, reason, source)` is the policy's final word.
- `Plan(steps: (PlanStep(description, capability_hint, inputs_hint, depends_on)…), rationale, revision)`.
