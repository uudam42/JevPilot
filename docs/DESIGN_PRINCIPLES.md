# JevPilot Design Principles

Every principle below says what it means, why it matters, and **how the
codebase enforces it**. An unenforced principle tends to erode.

## 1. Domain independence

The core contains no scientific or engineering domain knowledge. It knows
goals, states, capabilities, observations and evaluations, and never what they
*mean*. `Goal.parameters`, `Goal.success_criteria`, `Artifact.content` and
`Candidate.content` are opaque to it.

*Enforced by* `test_core_has_no_domain_vocabulary` and
`test_core_does_not_name_demo_domains`.

## 2. Dependency inversion

Domains depend on JevPilot's abstractions, and JevPilot never depends on
domains. The core loads domain modules by object, by `"module:Attr"` string, or
by entry point, and never through a static import.

*Enforced by* `test_core_never_imports_domains` and `test_layering`, which also
fixes the internal order core → interfaces → registry → orchestration.

## 3. Structured state over agent-to-agent conversation

Components communicate only through the typed, immutable `WorkflowState`, and
never by passing free-text messages to each other. Anything a later step needs
must be in state as an observation, artifact, candidate, context entry,
uncertainty record or evaluation.

*Enforced by* frozen Pydantic models with `extra="forbid"`, and tuples for
ordered collections.

## 4. Capabilities over hard-coded agents

The router chooses among *capabilities* described by metadata, and never
between named agents. An "agent" is just one kind of capability.
The routing problem is `(state, goal, available capabilities) → next capability`.

*Enforced by* `Router.select_next(state, capabilities: Sequence[CapabilitySpec])`.

## 5. Router/executor separation

The router decides and the executor executes. Routers receive `CapabilitySpec`
descriptors that have no `execute` method. This makes router benchmarking
honest: two routers can be compared on identical execution machinery.

*Enforced by* types, by `test_router_is_replaceable_same_outcome`, and by
`test_routers_only_ever_see_descriptors`. Model routers see an even narrower
view: a JSON `RoutingRequest`.

## 6. Explicit feedback loops

Each iteration runs Observe, then Update State, then Evaluate, then Control
decision, in that order. Retry, replan and escalation are explicit
`ControlAction` values recorded in `state.history` and in the trace. None of
them is hidden inside a component.

*Enforced by* `tests/integration/test_control_flow.py`.

## 7. Provenance-first scientific execution

The executor stamps every observation with provenance: capability, version,
domain, inputs, input digest, time window, external sources and upstream
`derived_from` links. It also stamps every artifact and candidate that lacks
provenance. `StateManager` refuses to admit un-provenanced artifacts or
candidates.

*Enforced by* `test_artifact_without_provenance_rejected` and the provenance-chain
assertion in `test_stats_domain.py`.

## 8. Replaceable routing policies

Rule-based, scripted, LLM, learned and Jev routers all implement the same
one-method interface. Nothing else in the system changes when the router
changes.

*Enforced by* `tests/routing/test_router_contract.py`. Rule, scripted, LLM, Jev
and fallback routers drive the same, unmodified `Controller`, `Executor` and
`StateManager` to the same outcome. The architecture test
`test_generic_orchestration_does_not_name_routing_implementations` keeps any
specific router or provider out of the generic loop.

## 9. Replaceable domain modules

Domains can be loaded, unloaded and combined at runtime. Switching domains
requires no change to the core.

*Enforced by* `tests/architecture/test_domain_replacement.py`. It covers two
unrelated demo domains, swapping via entry points, and an ad-hoc third domain
defined inside the test itself.

## 10. Testable state transitions

`S_{t+1} = Update(S_t, O_t)` is a pure function (`StateManager.update`) that
returns a new state and leaves its inputs untouched. Domain-specific
transitions are pure `StateReducer`s. Both can be unit-tested with no
controller, router or I/O.

*Enforced by* `tests/unit/test_state_manager.py`.

---

### Supporting rules

- **Recommend vs decide.** Evaluators recommend and a `ControlPolicy` decides,
  so termination rules never live in the controller.
- **Failures are data.** Capability errors become observations, not exceptions.
- **LLMs and Jev are adapters, never dependencies.** The core loop runs with
  no model. Provider SDKs live in `integrations/`, outside the core package,
  as optional extras. The architecture tests forbid SDK imports in the core,
  and forbid `routing` from importing concrete adapters.
- **Model output is untrusted.** It is parsed into a fixed JSON shape and
  validated against the offered capabilities and their input schemas, first
  by the router, then again by the controller and the executor. It can never
  execute, import or configure anything (`tests/routing/test_parsing.py`,
  `tests/integration/test_routing_failures.py`).
- **Failures are explicit, fallback is explicit.** Routing failures are
  traced and handed to the control policy, which by default stops the run.
  Falling back to another router happens only through a configured
  `FallbackRouter`, and every attempt is recorded
  (`tests/routing/test_fallback.py`).
- **Measure, don't assume.** Router confidence is stored and measured, never
  treated as calibrated. Unknown token counts or costs are recorded as
  unknown, never as zero or a guess.
- **Simplicity first.** One runtime dependency (Pydantic), synchronous
  execution, no services or queues. Complexity is added only when a later phase
  needs it.
