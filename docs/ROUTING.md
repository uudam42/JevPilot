# Routing

A router is a routing policy:

$$R(S_t, C_t) \rightarrow D_t$$

It takes the current workflow state $S_t$ and the descriptions of the
capabilities available right now, $C_t$, and returns a structured decision
$D_t$. Every router in JevPilot works this way: rule-based, scripted, LLM,
Jev, fallback chains, and future learned or human routers. The controller,
executor, state manager, registries and evaluators do not know which one is
in use.

```text
            WorkflowState + CapabilitySpecs (descriptions only)
                                 │
     ┌───────────┬───────────────┼───────────────┬──────────────────┐
     ▼           ▼               ▼               ▼                  ▼
 RuleRouter  ScriptedRouter   LLMRouter       JevRouter      FallbackRouter(…)
                                 │               │
                        RoutingRequest     RoutingRequest
                                 │               │
                        prompts.py (LLM)         │
                                 ▼               ▼
                            LLMAdapter   RoutingModelAdapter   ← providers live here
                                 │               │
                            raw text/JSON (untrusted)
                                 ▼
                     parsing.decision_from_output  (strict shape)
                                 ▼
                     validate_decision (offered id, input schema, plain data)
                                 ▼
                          RoutingDecision  ──►  Controller re-validates ──► Executor
```

## 1. The contract

`jevpilot/interfaces/router.py`

```python
class Router(ABC):
    router_id: str
    def select_next(self, state, capabilities: Sequence[CapabilitySpec]) -> RoutingDecision: ...
    def route(self, state, capabilities) -> RoutingOutcome        # decision + diagnostics
    def config(self) -> dict                                       # recorded for reproducibility
```

- `select_next` is the only method a router must implement.
- `route` is what the controller calls. By default it wraps `select_next` in
  one timed `RoutingAttempt`. Model-backed and composite routers override it
  to report the routing request, model usage and fallback attempts.
- Routers receive `CapabilitySpec` descriptors, never `Capability` objects. A
  router cannot execute anything.
- A router signals an explicit **routing failure** by raising a `RoutingError`
  subclass. Any other exception is an orchestration bug. The workflow is marked
  `FAILED` with an `error` trace event, which is unchanged from Phase 0.

## 2. RoutingDecision

`jevpilot/core/decision.py`

| Field | Meaning |
|---|---|
| `capability_id` | Capability to invoke, or `None` for no action |
| `inputs` | JSON inputs for that capability |
| `intent` | `invoke`, or one of three no-action intents: `finish` (goal met), `ask_human` (missing information only a human can give), `idle` (nothing to offer) |
| `reason` | Short justification |
| `confidence` | Self-reported, in [0, 1], or `None`. **Stored and measured, never trusted** |
| `alternatives`, `router_id`, `metadata` | Other options considered, which router decided, and extras (for example `metadata["fallback"]`) |

`intent` defaults from `capability_id`, so Phase 0 code that builds decisions
still works. `ask_human` is a legitimate outcome, not an error: the controller
ends the run in `AWAITING_HUMAN` with reason `router requests human input`.
Resuming with a human's answer is Phase 2 work.

## 3. Validation: what a decision must satisfy

`validate_decision(decision, capabilities)` in `interfaces/router.py` checks
that:

1. **the capability was offered.** It must be one of the specs passed to the
   router. A registered but inapplicable capability counts as not offered.
   Failure raises `InvalidCapabilityError`.
2. **the inputs are plain JSON data.** No objects, callables or non-string
   keys. Failure raises `InvalidRoutingInputError`.
3. **the inputs satisfy the capability's `input_schema`.** Types, required
   fields and `extra="forbid"` all apply. Failure raises
   `InvalidRoutingInputError`.

`ModelRouter` and `FallbackRouter` run it on every attempt. **The controller
runs it again on every decision from every router** before anything reaches
the executor, and the executor still validates inputs as in Phase 0. A
router therefore cannot silently produce a nonexistent capability, invalid
inputs, missing fields or a malformed decision.

## 4. Routing failures

`jevpilot/exceptions.py`. All of these are subclasses of `RoutingError`, which
carries `retryable`, `details` and `attempts`.

| Error | Raised when |
|---|---|
| `InvalidCapabilityError` | The decision names a capability that was not offered |
| `InvalidRoutingInputError` | The inputs are not plain data or fail the input schema |
| `MalformedRoutingDecisionError` | Model output is not a valid decision payload, or a router returned something other than a `RoutingDecision` |
| `RouterTimeoutError` | The adapter timed out, or answered after `timeout_s` |
| `RouterAdapterError` | The provider or adapter failed: network, refusal, truncation, or any SDK exception |
| `LowConfidenceRoutingError` | Confidence is below a configured `min_confidence`. This check is off by default |

There is no `NoAvailableCapabilityError`. When nothing is available, the
controller does not call the router at all and records a `no capabilities
available` no-action decision, as in Phase 0.

**Controller behaviour.** On a `RoutingError`, the controller:

1. emits a `routing_failure` trace event with the error, available
   capability ids, routing latency and every attempt;
2. appends a `HistoryEntry` whose `routing_error` holds the error (its
   `decision` is a no-action placeholder);
3. asks `policy.on_routing_failure(state, error)` what to do.

It executes nothing for that iteration. The base `ControlPolicy` terminates
with failure. `DefaultControlPolicy(max_routing_failures=n)` allows the router
to be asked again up to `n` consecutive times, and each re-route is a traced
`CONTINUE` decision. The evaluator still only recommends; the policy decides.

## 5. RoutingRequest

`jevpilot/routing/request.py`

`RoutingRequest.from_state(state, capabilities)` produces the canonical,
serialisable routing problem. Model adapters send it, fixtures store it and
replays reuse it.

| Field | Content |
|---|---|
| `schema_version` | `jevpilot.routing_request/1` |
| `goal` | description, parameters, success criteria |
| `state_summary` | status, context, **extensions** (fields added by a `WorkflowState` subclass), artifact index (id, name, kind; no content), candidates, uncertainty, plan steps |
| `requirements`, `constraints` | as structured data |
| `available_capabilities` | `CapabilityDescription`: id, name, description, version, domain, input/output **JSON Schema**, tags, preconditions, effects, side effects, cost and latency estimates (**only if the capability declared them**), requirements, metadata |
| `previous_actions` | last `max_actions` iterations: capability, intent, inputs, success, result, error, control action, whether it was a retry |
| `previous_evaluations` | last 3: recommendation, progress, gaps, unsatisfied constraints, rationale |
| `step`, `retry_count`, `metadata` | |

The request contains JSON data only. Large values are truncated
deterministically (`max_value_chars`). Provenance, timestamps and artifact
contents are left out. `to_json()` is canonical (sorted keys), and
`fingerprint()` gives a stable digest.

## 6. The routers

### RuleRouter: the baseline

It is deterministic, makes no model calls, and uses the first matching rule.
Rules (`Rule(capability_id, when, inputs, reason)`) are **supplied by the
domain or benchmark suite**, never built into the core. `config()` lists them.
If no rule matches, the router returns an `idle` no-action.

### LLMRouter

```python
LLMRouter(adapter: LLMAdapter, *, prompt_builder=build_routing_prompt,
          timeout_s=None, min_confidence=None, max_actions=20, trace_requests=True)
```

The pipeline is: `RoutingRequest` → `prompts.build_routing_prompt` (versioned
`routing-prompt/1`) → `adapter.complete(prompt)` → text → strict parse →
validate. All prompt text lives in `jevpilot/routing/prompts.py`. The prompt
tells the model to:

- choose only offered ids;
- respect the schemas;
- return exactly one JSON object;
- execute nothing;
- use the current state;
- avoid unnecessary actions;
- use `finish` or `ask_human` when appropriate.

### JevRouter

```python
JevRouter(adapter: RoutingModelAdapter, *, timeout_s=None, min_confidence=None, ...)
```

It sends the canonical `RoutingRequest` to `adapter.infer(request)`. Any
Jev-specific formatting and transport belong inside the adapter. The output
goes through the same parser and validator as the LLM path. **No part of the
controller, executor, state, registry, domain interface or evaluation knows
Jev exists** (enforced by `test_generic_orchestration_does_not_name_routing_implementations`).

### FallbackRouter: explicit fallback by composition

```python
FallbackRouter(JevRouter(...), RuleRouter(rules))                     # Jev → Rule
FallbackRouter(JevRouter(...), LLMRouter(...), RuleRouter(rules))     # Jev → LLM → Rule
FallbackRouter(primary, fallback, fallback_on=(MalformedRoutingDecisionError,))
```

- Routers never fall back on their own. Without a `FallbackRouter`, a routing
  failure goes straight to the policy, and by default the workflow stops.
- Only `RoutingError`s listed in `fallback_on` trigger the next router. The
  default is all of them. Router bugs propagate.
- A no-action decision from the primary is a decision, not a failure.
- The outcome lists **every** attempt: router, error, latency, model, usage and
  request. The decision carries `metadata["fallback"] = {primary_router,
  decided_by, failures}`. The controller's `routing_decision` event includes
  `fallback_used` and all attempts.
- If every router fails, the last error is raised with all attempts attached.

## 7. Adapters

Contracts are in `jevpilot/routing/contracts.py`. Adapters depend on them;
routing never imports an adapter.

| Protocol | Method | Used by |
|---|---|---|
| `LLMAdapter` | `complete(prompt: LLMPrompt, *, timeout_s) -> LLMCompletion(text, model, usage, stop_reason)` | `LLMRouter` |
| `RoutingModelAdapter` | `infer(request: RoutingRequest, *, timeout_s) -> RoutingModelResponse(output, model, usage)` | `JevRouter` |

Both also have `describe() -> dict`, which gives the provider, model and
sampling configuration and is recorded in run manifests. `ModelUsage`
(`model_calls`, `input_tokens`, `output_tokens`, `estimated_cost_usd`) uses
`None` for *unknown*. Adapters report only what the provider returned.

| Adapter | Where | Real or mocked |
|---|---|---|
| `FakeJevAdapter`, `FakeLLMAdapter` | `jevpilot/adapters/` | **Mocked.** Offline and deterministic. Driven by a script or a request-level policy |
| `AnthropicLLMAdapter` | `integrations/anthropic_llm.py` | **Real** Claude Messages API. Optional (`[anthropic]`), strict single-model by default. SDK-verified, not yet live-verified |
| `TypeSafeJevAdapter` | `integrations/typesafe_jev.py` | **Real** TypeSafe Jev (`system_one`). Optional (`[jev]`). SDK-verified, not yet live-verified; identity to be confirmed. See [REAL_ROUTING.md](REAL_ROUTING.md) |

Fake adapters accept script items built with `jevpilot.adapters.scripting`:
`valid`, `finish`, `ask_human`, `invalid_capability`, `invalid_inputs`,
`low_confidence`, `malformed`, `timeout`, `adapter_error`, plus `with_faults`
for seeded probabilistic fault injection. Adapters that need an SDK live
outside `jevpilot/`, so the core's no-SDK import guard stays intact.

## 8. Security boundary

Model output is untrusted input. By construction:

- the parser accepts exactly the keys `action`, `capability_id`, `inputs`,
  `reason` and `confidence`. Anything else (including an attempt to set
  `router_id`) is a `MalformedRoutingDecisionError`;
- nothing is evaluated, imported or instantiated from model output. JSON is
  decoded to plain data only, and the inputs are checked to be plain data
  again;
- capability ids resolve only against the offered specs, and then through
  the registry in the executor;
- inputs are validated against the capability's schema by the router, the
  controller and the executor, before invocation;
- a model cannot change configuration. The only thing it can influence is a
  validated `RoutingDecision`.

Tests: `tests/routing/test_parsing.py`,
`tests/integration/test_routing_failures.py::test_model_output_cannot_reach_beyond_its_decision`.

## 9. Adding another router without touching orchestration

1. **Rule-like or learned policy:** subclass `Router` and implement
   `select_next`. Override `config()` so runs record your parameters.
2. **Model-backed policy:** subclass `ModelRouter`. Implement
   `_infer(request) -> RoutingModelResponse` and `adapter_config()`. You get
   request building, parsing, validation, timeouts, confidence thresholds and
   attempt tracing for free.
3. **A real Jev adapter:** implement `RoutingModelAdapter`:

   ```python
   class JevHttpAdapter:
       def infer(self, request: RoutingRequest, *, timeout_s=None) -> RoutingModelResponse:
           payload = to_jev_format(request)             # Jev-specific, stays here
           raw = call_jev(payload, timeout=timeout_s)   # transport, auth from env
           return RoutingModelResponse(output=raw_decision_json(raw), model=..., usage=...)
       def describe(self) -> dict: return {"provider": "jev", "model": ..., ...}

   router = JevRouter(JevHttpAdapter())
   ```

   Put it outside `jevpilot/` (for example `integrations/jev_*.py`) if it
   needs third-party packages.
4. **Human router** (future): a `Router` that returns `ask_human` or reads a
   queued answer. Nothing else changes.
5. Add the router to `experiments/routing/routers.py` to benchmark it.

No change to the `Controller`, `Executor`, `StateManager`, registries,
domains or evaluators is needed at any step.
