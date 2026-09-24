# JevPilot Architecture

JevPilot is an **orchestration system**, not a collection of agents. Agents,
tools, models, simulators, databases and optimisers are all represented as
**capabilities** that a pluggable **router** selects and a separate **executor**
runs, over one structured **workflow state**.

## 1. Layers

```text
┌──────────────────────────────────────────────────────────────────────┐
│ JevPilot core (domain-agnostic)                                      │
│                                                                      │
│  orchestration  Controller · Executor · StateManager · Policy · Runtime│
│        │                                                             │
│  registry       CapabilityRegistry · DomainRegistry                  │
│        │                                                             │
│  interfaces     Capability · Router · Planner · Evaluator ·          │
│                 ControlPolicy · StateReducer · DomainModule          │
│        │                                                             │
│  core           WorkflowState · Observation · Artifact · Provenance ·│
│                 Uncertainty · EvaluationResult · RoutingDecision ·   │
│                 Plan · Trace                                         │
│                                                                      │
│  routing        RoutingRequest · RuleRouter · ScriptedRouter ·       │
│                 LLMRouter · JevRouter · FallbackRouter · prompts ·   │
│                 adapter contracts (LLMAdapter, RoutingModelAdapter)  │
│        │                                                             │
│  adapters       offline fakes (FakeLLMAdapter, FakeJevAdapter)       │
│  planning       NullPlanner · StaticPlanner                          │
└──────────────────────────────▲───────────────────────────────────────┘
                               │ imports public API only
┌──────────────────────────────┴───────────────────────────────────────┐
│ Domain modules: capabilities, evaluators, state extensions, reducers, │
│ routing heuristics, schemas, models, data                            │
│   domains/demo (arithmetic)  domains/stats_demo  domains/uav_materials│
├──────────────────────────────────────────────────────────────────────┤
│ integrations/  SDK-backed provider adapters (Claude, TypeSafe Jev)   │
│ experiments/   routing benchmark: fixtures, suites, metrics, CLI     │
│ apps/          end-to-end applications: composition roots + CLI      │
└──────────────────────────────────────────────────────────────────────┘
```

### Dependency direction

| Package | May import (inside `jevpilot`) |
|---|---|
| `core` | nothing |
| `interfaces` | `core`, `exceptions` |
| `registry` | `core`, `interfaces`, `exceptions` |
| `orchestration` | `core`, `interfaces`, `registry`, `exceptions` |
| `routing` | `core`, `interfaces`, `exceptions` |
| `adapters` | `core`, `interfaces`, `routing`, `exceptions` |
| `planning` | `core`, `interfaces` |
| `domains/*`, `integrations/*`, `experiments/*` | any public `jevpilot` name |
| `apps/*` | public `jevpilot` names, `jevpilot.routing`, `jevpilot.adapters`, `domains/*`, `integrations/*` |

`jevpilot` → `domains`, `integrations`, `experiments` is **forbidden**, and so
is `routing` → `adapters`: adapters implement routing contracts, never the
reverse. `tests/architecture/test_dependency_direction.py`
enforces this table by parsing every core file's imports. It also rejects LLM or
heavyweight framework SDK imports and domain vocabulary (for example
"corrosion" or "aerospace") anywhere in core source. It also checks that
`core`, `interfaces`, `registry`, `orchestration` and `planning` code never
names a routing implementation or provider (`JevRouter`, `LLMRouter`,
`RoutingRequest`, adapters, …). Provider SDKs therefore live in
`integrations/`, outside the core package.

Domains reach the core without static imports, in one of three ways:
`Runtime.load(obj)`, `Runtime.load("pkg.module:Attr")`, or an installed entry point
in group `jevpilot.domains`.

## 2. The control loop

`jevpilot/orchestration/controller.py`:

```text
start → [plan] ─┐
                ▼
   capabilities = registry.available(state)        # preconditions filter
   outcome      = router.route(state, specs)       # π(S_t) → A_t   (decides)
   validate_decision(outcome.decision, specs)      # offered id, schema-valid inputs
   observation  = executor.execute(decision, state)# A_t → O_t      (acts)
   state        = state_manager.update(S_t, A_t, O_t)  # S_{t+1}
   evaluation   = evaluator.evaluate(state)        # domain-defined "good"
   control      = policy.decide(state, evaluation, observation)
                │
     CONTINUE ──┤ loop
     RETRY ─────┤ re-execute the same decision (no re-routing)
     REPLAN ────┤ planner.plan(state) → state.plan (revision+1), loop
     TERMINATE_SUCCESS / TERMINATE_FAILURE / HUMAN_INTERVENTION → stop
```

Special cases, all domain-neutral:

- **The router returns no action** (`capability_id=None`), or no capability is
  available. The controller evaluates the state. If the evaluator says the goal
  is met (or failed), that result stands. Otherwise the controller escalates to
  `HUMAN_INTERVENTION`.
- **Capability failure.** The executor never raises. Failures become
  `Observation(success=False, error=ErrorInfo(retryable=…))` and the policy
  decides what happens next.
- **Routing failure.** If the router raises a `RoutingError` or its decision
  fails validation, nothing is executed. The controller emits
  `routing_failure`, records a `HistoryEntry` with `routing_error`, and asks
  `policy.on_routing_failure(state, error)`. By default this terminates the
  run; `DefaultControlPolicy(max_routing_failures=n)` allows bounded
  re-routing. Fallback between routers is explicit (`FallbackRouter`). See
  [ROUTING.md](ROUTING.md).
- **Human input.** A decision with intent `ask_human` ends the run in
  `AWAITING_HUMAN` as a legitimate outcome.
- **Orchestration bug.** If a router (with a non-`RoutingError` exception), evaluator or reducer raises, the workflow
  is marked `FAILED`, an `error` trace event is emitted, and the result carries
  the `ErrorInfo`.
- **Safety cap.** The controller stops after `max_iterations` (default 1000),
  whatever the policy says.

## 3. Core abstractions

| Abstraction | Module | Role |
|---|---|---|
| `WorkflowState` | `core/state.py` | S_t: goal, context, requirements, constraints, observations, artifacts, candidates, evaluations, uncertainty, provenance, plan, history, status |
| `CapabilitySpec` / `Capability` | `interfaces/capability.py` | Router-facing metadata / executable object |
| `CapabilityRegistry` | `registry/capability_registry.py` | `register`, `unregister`, `get`, `list`, `find`, `available(state)` |
| `Router` → `RoutingDecision` | `interfaces/router.py`, `core/decision.py` | π(state, specs) → structured decision; `route()` adds `RoutingOutcome` diagnostics |
| `RoutingRequest` | `routing/request.py` | Canonical JSON routing problem for model routers, fixtures, replay |
| `LLMRouter`, `JevRouter`, `FallbackRouter` | `routing/` | Adapter-backed and composite routers ([ROUTING.md](ROUTING.md)) |
| `Executor` | `orchestration/executor.py` | Resolve → preconditions → validate input → execute → validate output → `Observation` |
| `StateManager` + `StateReducer` | `orchestration/state_manager.py`, `interfaces/reducer.py` | Pure transition S_{t+1} = Update(S_t, O_t) |
| `Evaluator` → `EvaluationResult` | `interfaces/evaluator.py`, `core/evaluation.py` | Domain judgement of progress and success |
| `ControlPolicy` → `ControlDecision` | `interfaces/policy.py`, `orchestration/policies.py` | Final loop action with generic budgets |
| `Planner` → `Plan` | `interfaces/planner.py`, `core/plan.py` | Advisory decomposition, independent of the router |
| `DomainModule` | `interfaces/domain.py` | Bundle of capabilities, evaluators, reducers and state type |
| `Tracer` / `TraceEvent` | `core/trace.py` | Structured trace to in-memory, JSONL or logging sinks |

### Capability model

A capability is *anything the system can do*. The core sees only its
`CapabilitySpec`:

```text
id (e.g. "<domain>.<name>") · name · description · version · domain
input_schema / output_schema (Pydantic models → JSON Schema via describe())
tags · preconditions (text) · effects (text) · side_effects
cost_estimate · latency_estimate · requirements · metadata
```

Executable preconditions live in `Capability.is_applicable(state)`. Execution
returns a `CapabilityResult` holding the output, artifacts, declarative
`StateEffects`, uncertainty, sources and `derived_from` links. It can also
return a bare value.

**Routers receive specs, never `Capability` objects.** A router therefore cannot
execute anything, and the router/executor separation holds by construction.
`spec.describe()` produces a JSON-serialisable view for LLM or Jev routers.

### Planner vs router

The planner maps goal and state to a `Plan`, which is stored in `state.plan`. The
router maps state and available capabilities to the next action, and may consult
`state.plan` or ignore it. The two are wired independently in `Controller`, so
any pairing works: an LLM planner with a Jev router, a rule planner with a Jev
router, and so on.

### Evaluation vs control

Evaluators *recommend* an action, and a `ControlPolicy` *decides*. The
`DefaultControlPolicy` follows terminal recommendations first. It then applies
generic budgets: `max_steps`, `max_retries` (retryable failures only),
`max_consecutive_failures` and `max_replans`. Changing termination behaviour
means swapping the policy or the evaluator. The controller never changes.

## 4. Domain extension model

See [DOMAIN_INTERFACE.md](DOMAIN_INTERFACE.md). In summary, a domain provides
capabilities, and optionally evaluators, reducers, a `WorkflowState` subclass
and routing heuristics. `Runtime.controller(router, domains=[...])` scopes which
domains' evaluators and reducers apply to a run, while every registered
capability stays visible to the router (cross-domain workflows remain possible).

Two unrelated demo domains prove the boundary
(`tests/architecture/test_domain_replacement.py`):

| | `domains/demo` (arithmetic) | `domains/stats_demo` |
|---|---|---|
| State | `NumberState(WorkflowState)` subclass | base `WorkflowState` + `context` |
| State update | `StateReducer` | generic `StateEffects` |
| Outputs | typed values + candidate | dataset/report **artifacts** |
| Uncertainty | certain | 95% interval, measurement kind |
| Provenance | per step | chained: report → summary → dataset → generator source |
| Failure | none | domain-defined sample budget |

Both run on the same `Controller`, `Executor`, `StateManager` and registries,
with no change to core code.

## 5. Tracing

Every run records, in order:

```text
workflow_started, state_snapshot, [plan_created],
(routing_decision, observation, state_snapshot, evaluation, control_decision
 | routing_failure, control_decision)*,
[error], workflow_completed
```

`workflow_started` carries `router_config` (`router.config()`: router type,
adapter, model, prompt version, rules, fallback chain). `routing_decision`
carries the decision, `routing_latency_s`, `available_capabilities`,
`fallback_used` and every `RoutingAttempt`. Each attempt holds the router,
success or error, latency, model, `ModelUsage`, and the serialised
`RoutingRequest` for model routers. `routing_failure` carries the structured
error and the same attempt records. With the state snapshots, observations
(including `execution_time`), evaluations and control decisions, a run can be
reconstructed step by step: state → routing request → decision → execution →
observation → transition → evaluation → control. Raw model output appears
only as a 300-character excerpt in malformed-output errors, and credentials
never pass through JevPilot.

Events carry `seq`, `workflow_id`, `step`, `timestamp` and a JSON payload.
`WorkflowResult.trace` always holds the in-memory copy. Pass
`trace_sinks=[JsonlTraceSink(path)]` or `LoggingTraceSink()` for persistent or
logged traces, and use `read_jsonl_trace` to reload one.

## 6. Known limitations and deliberate simplifications

- Execution is synchronous and sequential: one capability per step.
- A workflow in `AWAITING_HUMAN` stops. There is no resume-with-human-input API yet.
- Router timeouts are soft deadlines: the adapter receives `timeout_s`, and
  late answers are discarded. Nothing is cancelled mid-call.
- The real provider adapters (`AnthropicLLMAdapter`, `TypeSafeJevAdapter`) are
  verified through their SDKs with mocked transports; live calls need
  credentials and have not been run in the development environment.
- `ScriptedRouter` is stateful (use one instance per run).
- Full state snapshots in the trace are simple but can be large.
- Demo domains ship in the same distribution as the core, for convenience.

## 7. Applications: the end-to-end layer

An *application* (`apps/`) is a composition root: it chooses the router, the
model adapters and the domain, and exposes one entry point. The UAV materials
application wires the `uav_materials` domain to either live providers (Claude
for requirement interpretation, Jev for routing) or offline stand-ins, runs
the unmodified controller loop, and returns a structured report plus
Markdown:

```python
from apps.uav_materials import run_uav_material_workflow

result = run_uav_material_workflow()  # OFFLINE_DEMO; mode="LIVE" needs API keys
```

The domain never imports routers, adapters or provider SDKs (a domain test
enforces it); the application does, so provider choice stays outside both the
core and the domain. `jevpilot uav-materials --demo` is the command-line
entry point. See [UAV_MATERIALS.md](UAV_MATERIALS.md) for the methodology.

## Phase 1 (done)

Routing layer and benchmarks: `RoutingRequest`; `LLMRouter`, `JevRouter` and
`FallbackRouter` behind adapter contracts; offline fake adapters with fault
injection; strict parsing and validation; explicit routing failures handled by
policy; routing diagnostics in the trace; a real Claude adapter in
`integrations/`; and the benchmark harness in `experiments/routing/`. See
[ROUTING.md](ROUTING.md) and [BENCHMARKING.md](BENCHMARKING.md).

## Phase 1.5 (done)

Real adapters for Claude (strict single-model mode by default) and TypeSafe
Jev in `integrations/`, verified through their SDKs. The routing
generalization benchmark: `benchmarks/routing/` data and the
`experiments/routing/generalization/` oracle, runner and metrics, with
offline and live modes. Git history and version 0.1.0 (experimental). See
[REAL_ROUTING.md](REAL_ROUTING.md), [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md)
and [EXPERIMENTS.md](EXPERIMENTS.md). No core module changed except
`ModelRouter`, which now keeps adapter metadata on every `RoutingAttempt`.

## UAV materials iterations and productization (done)

A real domain on the unchanged core: real material data with provenance,
requirement semantics per material system, evidence-aware evaluation,
NASA-sourced micromechanics and lamination theory, bounded inverse design, a
configurable decision policy, the end-to-end application with offline and
live modes, and report-integrity tests. See [UAV_MATERIALS.md](UAV_MATERIALS.md).

## Phase 2 backlog

1. Live runs: the smoke tests, then `--mode live` on validation and eval with
   repetitions, once credentials (and confirmation of Jev's identity) exist.
   Then a cache-friendly request layout (stable capability list before the
   volatile state) to cut live cost; that needs a new prompt version.
2. Human-in-the-loop resume (`Controller.resume(state, human_input)`) and a
   `HumanRouter`.
3. `HybridRouter` (for example rules for known states, a model otherwise) and
   learned routers trained on logged `RoutingRequest` → decision pairs.
4. Confidence-calibration analysis over benchmark records.
5. LLM adapters for planning and reporting behind `Planner` and new
   interfaces (requirement interpretation exists for the UAV application).
6. Async and parallel capability execution, hard timeouts and cancellation.
7. A persistent provenance store and artifact storage (URIs, content hashing).
8. Richer uncertainty propagation and conflicting-observation handling.
9. State-snapshot diffs instead of full snapshots, and a trace replay tool.
10. Moving demo domains into separate distributions; the first real domain package.
