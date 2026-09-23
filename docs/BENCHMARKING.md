# Routing Benchmarks

`experiments/routing/` compares routing policies on **identical states and
capabilities**: same workflow, same initial state, same registry, different
router. It uses artificial or demo domains only; no specialist domain
knowledge is involved.

```bash
python -m experiments.routing.benchmark                            # all simulated routers, both modes
python -m experiments.routing.benchmark --router rule --router jev+rule
python -m experiments.routing.benchmark --mode decision --repeats 5 --seed 7
python -m experiments.routing.benchmark --no-faults                # fakes without fault injection
python -m experiments.routing.benchmark --router llm-anthropic     # REAL API calls (credentials, cost)
```

Results go to `experiments/routing/results/<run_id>/`, which is git-ignored.
Use `--no-write` to print the summary only.

## 1. Two modes

| Mode | Unit | What runs |
|---|---|---|
| `decision` | one routing step, $R(S_t, C_t) \rightarrow D_t$ | `router.route()` on a fixture state, then `validate_decision` exactly as the controller does |
| `workflow` | one whole workflow | the unmodified `Controller` with `DefaultControlPolicy(max_steps=30)`; metrics are read from the trace |

A fresh router is built for every case and repeat, so no state leaks between
cases.

## 2. Fixture format

### Decision cases (`fixtures/decision_cases.json`)

```jsonc
{
  "benchmark": "jevpilot.routing.decisions",
  "version": "1",
  "suite": "pipeline",                      // whose rules/policy the baseline routers use
  "capability_catalog": {                   // reusable capability descriptions
    "pipe.fetch": {"description": "...", "input_schema": {JSON Schema}, "tags": [...],
                   "preconditions": [...], "effects": [...], "side_effects": false}
  },
  "cases": [{
    "case_id": "retry_01_transient_fetch",
    "pattern": "retry_after_failure",
    "description": "...",
    "state": {
      "goal": {"description": "...", "parameters": {...}, "success_criteria": {...}},
      "context": {...},
      "artifacts": [{"id": "art_raw", "name": "raw_records", "kind": "dataset"}],
      "history": [{"capability_id": "pipe.fetch", "inputs": {...}, "success": false,
                   "error": {"type": "...", "message": "...", "retryable": true},
                   "control": "continue"}],
      "evaluations": [{"recommendation": "replan", "goal_progress": 0.4, "remaining_gaps": [...]}]
    },
    "capabilities": ["pipe.fetch", "pipe.fetch_mirror", {"id": "x.inline", ...}],
    "expected": {
      "valid_capabilities": ["pipe.fetch", "pipe.fetch_mirror"],     // any of these
      "forbidden_capabilities": [],
      "unnecessary_capabilities": ["pipe.cleanup_cache"],
      "required_inputs": {"pipe.fetch": {"source": "..."}},          // subset match
      "no_action_intents": []                                       // e.g. ["finish"], ["ask_human", "idle"]
    }
  }]
}
```

- `capabilities` are the ones **offered at that step**. Preconditions are
  expressed by leaving capabilities out.
- Input schemas are JSON Schema. `cases.model_from_json_schema` turns them
  into Pydantic models, so validation is identical to code-defined
  capabilities. It supports types, `enum`, `minimum`/`maximum`, `required` and
  `additionalProperties: false`.
- The compact `state` is expanded into a real `WorkflowState` (observations,
  history and evaluations included) by `cases.build_state`.
- **Several answers may be acceptable.** A decision is acceptable if it
  invokes any `valid_capabilities` entry whose inputs contain the
  `required_inputs` for it, or if it proposes no action with an intent in
  `no_action_intents`.

Covered patterns: sequential dependency, missing information, capability
preconditions, redundant capability, invalid capability, multiple valid
capabilities, retry after failure, replan after observation, termination.

### Workflow cases (`fixtures/workflow_cases.json`)

```json
{"case_id": "pipe_03_flaky_source", "suite": "pipeline", "pattern": "retry_after_failure",
 "params": {"source": "...", "flaky": true}, "expected_status": "succeeded",
 "optimal_steps": 6, "unnecessary_capabilities": ["pipe.cleanup_cache"]}
```

`suite` selects an executable suite in `experiments/routing/suites/`:

| Suite | Domain | Notes |
|---|---|---|
| `arithmetic` | Phase 0 `domains/demo`, unchanged | state subclass, reducer |
| `stats` | Phase 0 `domains/stats_demo`, unchanged | artifacts, uncertainty, re-sampling |
| `pipeline` | `suites/pipeline.py`, benchmark-local | fetch → parse → validate → summarize → publish, with a mirror (redundant), a cache cleaner (unnecessary), a flaky source (retry), dirty data (replan and clean), and missing source (human) |

Each suite supplies its **rules** (the `RuleRouter` baseline) and a
**request-level policy** that sees only the JSON `RoutingRequest`. That policy
drives the simulated model routers. Routing heuristics live with the suite,
never in the core.

## 3. Routers

| Name | Router | Real or simulated |
|---|---|---|
| `rule` | `RuleRouter(suite.rules())` | deterministic baseline |
| `jev` | `JevRouter(FakeJevAdapter(policy=with_faults(suite.policy, JEV_FAULTS)))` | **simulated** |
| `llm` | `LLMRouter(FakeLLMAdapter(policy=with_faults(suite.policy, LLM_FAULTS)))` | **simulated** (the full prompt → text → parse path) |
| `jev+rule`, `llm+rule`, `jev+llm+rule` | `FallbackRouter(...)` | simulated primaries |
| `llm-anthropic` | `LLMRouter(AnthropicLLMAdapter())` | **real** |

> **Read simulated numbers correctly.** The fake adapters answer with the
> suite's reference policy, and seeded fault injection replaces some answers
> with malformed output, unknown capabilities, bad inputs, timeouts or
> provider errors. The rates are arbitrary parameters in `routers.py`.
> Simulated results measure the **harness, validation and fallback
> machinery**. They say nothing about how Jev or any LLM routes. No claim
> about the relative quality of routers should be drawn from them.

## 4. Metrics

### Decision mode (`metrics.decision_metrics`)

| Metric | Definition |
|---|---|
| Valid decision rate (VDR) | valid final decisions / decisions |
| Routing accuracy (RA) | acceptable decisions / decisions (also reported *given valid*) |
| Forbidden / unnecessary selection rate | final decisions invoking a forbidden / unnecessary capability |
| No-action rate | valid decisions proposing no action |
| Fallback rate | decisions where at least one attempt failed before a router succeeded |
| Invalid capability rate (ICR) | attempts failing with `InvalidCapabilityError` / attempts |
| Invalid input, malformed, timeout, adapter error rates | the same, per error type |
| Routing latency | ms per decision: mean, median, min, max, p95 |
| Confidence | reported count and mean confidence for acceptable vs. not acceptable decisions |
| Usage | summed tokens, calls and cost; `null` if no attempt reported them |

Attempt-level rates include failed primaries hidden behind a fallback. For a
single router, attempts equal decisions and ICR matches its textbook
definition.

### Workflow mode (`metrics.workflow_metrics`)

| Metric | Definition |
|---|---|
| Task completion rate (TCR) | runs whose final status equals `expected_status` / runs (`awaiting_human` is correct for missing-information cases) |
| Steps to completion | executed capabilities in completed runs: mean, median, range, histogram |
| Excess steps | steps − `optimal_steps` |
| Unnecessary calls | executions of the case's `unnecessary_capabilities` |
| Routing failure rate | routing failures / router calls |
| Fallback rate | routed decisions that needed fallback / routed decisions |
| Retry, replan rate | `retry` / `replan` control decisions per executed step |
| **Routing latency** | per decision and per workflow, from the `routing_decision` / `routing_failure` events |
| **Execution latency** | per capability call and per workflow, from `observation.execution_time` |
| Total latency | wall-clock `Controller.run` time |
| Usage | as above |

Routing and execution latency are measured separately and never mixed.
Re-executions from a retry involve no router call and are not counted as
routing decisions.

## 5. Output files

| File | Content |
|---|---|
| `manifest.json` | run id, timestamp, benchmark version, seed, repeats, fault switch, per-router kind, fault rates and `config()` for each suite (adapter, model, prompt builder, timeouts, rules), fixture digests, case ids, JevPilot and Python versions, platform, `real_model_calls` flag |
| `decisions.jsonl` | one record per case × repeat × router: selected capability, intent, inputs, validity, error type, acceptable / forbidden / unnecessary, confidence, latency, fallback, attempt routers and errors, models, usage |
| `workflows.jsonl` | one record per case × repeat × router: status vs. expected, steps, capability sequence, retries, replans, routing failures, fallbacks, routing and execution latency lists, usage, final control reason |
| `summary.json` | all aggregate metrics above, per mode and router |
| `summary.csv` | headline metrics, one row per mode × router |

## 6. Reproducibility

- Per-case seeds derive deterministically from `--seed`, the case id and the
  repeat. With the same seed and fixtures, simulated runs reproduce every
  record exactly, apart from latency fields and the run id
  (`tests/benchmark/test_benchmark.py::test_same_seed_reproduces_results_exactly`).
- Model-backed routing is **not** assumed to be deterministic. Use
  `--repeats N` and compare distributions.
- Fixture files carry `version`, and the manifest stores their digests.

## 7. Interpreting results

- Compare routers **within one run**: same fixtures, seed and framework
  version.
- A low VDR together with a high fallback rate means the primary fails often
  but the chain recovers. Check `attempt_errors` to see why.
- A high VDR with low RA means the router makes valid but wrong choices.
  Inspect `decisions.jsonl` by `pattern`.
- Confidence is stored for calibration studies. Do not read it as a
  probability of correctness.
- Always check `manifest.json → routers → kind` before quoting a number, to
  see whether it came from a simulated router.

## 8. Adding cases, suites and routers

- **Decision case:** add an entry to `decision_cases.json`. Reuse catalog
  capabilities or inline new ones.
- **Workflow case:** add an entry to `workflow_cases.json` for an existing
  suite.
- **Suite:** add a `Suite` in `experiments/routing/suites/__init__.py` with a
  domain factory, state builder, rules and request policy.
- **Router:** add a branch to `routers.make_router`. For a new model router,
  see [ROUTING.md §9](ROUTING.md#9-adding-another-router-without-touching-orchestration).
