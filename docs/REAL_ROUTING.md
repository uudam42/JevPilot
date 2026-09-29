# Real Routing Integrations

JevPilot routes through adapters. The core never imports a provider SDK, and
real integrations live in `integrations/` as optional extras. This page
records what is real, how each integration was verified, and what is still
missing.

| Router | Adapter | Status (0.1.0) |
|---|---|---|
| `RuleRouter` | none | real, deterministic |
| `LLMRouter` | `integrations/anthropic_llm.py` (`AnthropicLLMAdapter`) | real code against `anthropic` 1.8.0; **verified through the SDK with a mock transport; no live call made yet** |
| `JevRouter` | `integrations/typesafe_jev.py` (`TypeSafeJevAdapter`) | real code against `typesafe-sdk` 0.7.1; **validated against the live API on 2026-09-29** (section 7); see [identity caveat](#is-this-the-intended-jev) |
| `LLMRouter` / `JevRouter` | `jevpilot/adapters` fakes | mocked, offline, infrastructure tests only |
| (not a router) | `integrations/langchain_chat.py` + `apps/uav_materials/langchain_interpreter.py` | optional LangChain layer for **requirement interpretation only**; **validated live with Claude on 2026-09-29** (section 8) |

Install: `pip install -e '.[anthropic]'`, `'.[jev]'` and/or `'.[langchain]'`.
Credentials come from the environment only (see `.env.example`). Nothing
secret is ever written to manifests or traces.

## 1. Strict experimental mode vs production mode

| | Strict (default) | Production |
|---|---|---|
| Purpose | attribution: a result labelled *model X* comes from model X | completion |
| Claude | plain `messages.create`; the served model must equal the requested model (or a dated snapshot of it), otherwise `RouterAdapterError` | `beta.messages.create(..., fallbacks="default")`: a policy-declined request may be re-run on another model; the served model is recorded |
| Router fallback | none: a routing failure goes to the control policy | allowed only as an explicitly labelled *system* (`jev>rule` and similar) |
| Benchmark flag | `--strict` (default) | `--no-strict`, recorded in the manifest |

The main Rule vs LLM vs Jev comparison never uses fallback. Fallback chains
run only in the `fallback` experiment and are reported as systems.

## 2. Claude (`AnthropicLLMAdapter`)

- **Model:** the `model` argument, else `JEVPILOT_ANTHROPIC_MODEL`, else
  `claude-opus-5`. The benchmark flag is `--llm-model`.
- **Preflight:** `verify()` calls `models.retrieve(model)`, so a live run fails
  fast (router marked *unavailable*) if the id does not exist for the account.
- **Request:** system prompt from `jevpilot/routing/prompts.py`
  (`routing-prompt/1`) plus the canonical `RoutingRequest` as JSON.
  `max_tokens=16000`. Effort and thinking are provider defaults unless
  `--llm-effort` is set. No sampling parameters are sent, because current
  models reject them.
- **Output:** text is parsed strictly as the JSON decision payload. Structured
  outputs (`output_config.format`) are not used, because the decision's
  `inputs` object is free-form; malformed output is measured as a routing
  failure.
- **Recorded per call:** provider, requested model, **served model** (from the
  response; never inferred), adapter version, mode, request id, stop reason,
  input and output tokens, and cache token counts.
- **Failure mapping:** timeout → `RouterTimeoutError`. Refusal, truncation, 4xx
  (not retryable) and 429/5xx (retryable) → `RouterAdapterError`.
- **Cost:** recorded only with `--llm-price in,out` (USD per MTok) plus
  `--pricing-source`. Otherwise `null`.

## 3. Jev

### Findings (inspected 2026-09-23; API contract re-checked 2026-09-29)

No Jev implementation, package or credential existed in this environment.
A public Jev was found and inspected from its official artefacts, and only
from those:

| Item | Finding (source: `typesafe-sdk` 0.7.1 source code and PyPI docs) |
|---|---|
| What it is | **Jev**, TypeSafe's "System One" model ([typesafe.ai](https://typesafe.ai), SDK repo `typesafe-ai/typesafe-sdk-python`) |
| Invocation | `TypeSafeClient(...).system_one(state, questions, model=..., timeout=...)`, which is `POST https://api.typesafe.ai/v1/systemone` |
| Input | `state`: text or JSON. `questions`: named `Choice` (≤ 255 labels with descriptions), `Noul` (yes/no), `Score` (ordered rubric) |
| Output | per question: `Choice` → `choice`, `confidence`, full `probabilities`; `Noul` → probability of yes; `Score` → expected score, confidence. Plus `model` (served) and `usage` (input/output tokens; output documented as free) |
| Models | `GET /v1/models` (`client.models.list()`) returns name, description, release date. The docs list `jev-1.13.0`, with the alias `jev-latest` pointing to it. On 2026-09-29 the API listed `jev-latest` and `jev-preview` ("a preview version of `jev-latest`"), and reported `jev-1.13.0` as the served model for `jev-preview` |
| Auth | `TYPESAFE_API_KEY` (also `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`) |
| Determinism controls | none exposed |
| Errors | typed; `TypeSafeAPITimeoutError` subclasses `TimeoutError`; HTTP status on `.status`; the docs name 401, 422, 429 and 529 and ask for backoff on 429/529 |
| SDK vs HTTP docs | agree: `Authorization: Bearer <key>`, body `{state, model, questions}`, same endpoints (SDK 0.7.1 source vs docs.typesafe.ai API reference) |
| Key property | **Jev generates no text.** It can only select among labels or values you give it |

### Mapping (`jevpilot-jev-questions/1`)

1. **Next action.** One `Choice`: the labels are the offered capability ids
   plus `__finish__` and `__ask_human__`. Each label's criterion is that
   capability's full description (description, input/output schema,
   preconditions, effects, tags, cost). The rest of the `RoutingRequest` is
   the `state`.
2. **Inputs** (only if the chosen capability has required fields; a second
   call):
   - `enum` → `Choice` over the enum values;
   - `boolean` → `Noul`;
   - bounded `integer` → `Score`;
   - anything else → `Choice` over the scalar values found in the state
     (goal parameters, context, state extensions), plus `__none__`.
3. The result becomes the standard decision payload and goes through the
   **same parser and validator** as every other model router.

Recorded per call: confidence, the full next-action probability
distribution, the input answers, requested and served model, calls (1–2)
and tokens.

### Unavoidable differences from the LLM route

- **Values.** Jev cannot write a value that is not already in the state. The
  benchmark's inputs are always grounded in goal parameters, so this is not
  a disadvantage *there*, but it would be for tasks that need a synthesised
  value.
- **Reasons.** Jev returns no reason text. `reason` is empty.
- **Calls.** Jev uses one call per decision, or two when inputs are needed. The
  LLM route uses one. Latency and usage are compared per decision.
- **Layout.** The information is the same, but its layout differs: capability
  descriptions become label criteria instead of prompt JSON. The underlying
  `RoutingRequest` is identical for both routes.

### Is this the intended Jev?

JevPilot's own documentation never defined "Jev". TypeSafe's Jev matches the
role closely: a decision model with calibrated confidence, which its own
docs describe as suited to "confidence-gated routing". **The project owner
must confirm the identity.** If "Jev" means something else, delete
`integrations/typesafe_jev.py`. Nothing else depends on it: `JevRouter` accepts
any `RoutingModelAdapter`.

### Model, reliability and diagnostics (`typesafe-jev-adapter/2`)

- **Model.** The `model` argument, else `TYPESAFE_DEFAULT_MODEL`, else discovered:
  the adapter lists the models for the credentials and pins the most recently released
  one. Nothing is hard-coded. `verify()` (the preflight) authenticates, resolves the model
  and checks that it is listed.
- **Retries.** SDK retries are off; the adapter retries itself so each event is
  recorded (`transport_events`). Rate limits (429, honouring `Retry-After`), timeouts,
  connection errors and 408/5xx (including 529) are retried at most `max_retries`
  (default 2) times with exponential backoff, and never past the router deadline.
  Authentication (401), permission (403), not-found (404) and invalid-request (400/422)
  errors fail immediately with a clear message.
- **Malformed answers.** A response without a well-formed `next_action` choice is a
  non-retryable `malformed_response`. Confidence above 1 by float round-off is
  clamped; anything else outside [0, 1] is rejected by the decision parser.
- **Secrets.** The key is read by the SDK from `TYPESAFE_API_KEY` only. Every error
  message passes through `integrations/redaction.py`; headers are never stored.
- **Diagnostics.** `UAVWorkflowResult.routing_log()` (CLI: `--routing-log FILE`) keeps
  only timestamp, step, router, model, intent, capability, confidence, latency_ms,
  status and retries.

### Running the live validation

Needs `TYPESAFE_API_KEY` in the environment. Cases were frozen before any live run
(`benchmarks/routing/samples/`), and the main benchmark never uses a fallback:

```bash
python -m experiments.routing.real_jev preflight                    # auth + model
python -m experiments.routing.benchmark --mode live --routers rule,jev \
    --sample tiny_live_v1 --experiments main                        # A: smoke
python -m experiments.routing.benchmark --mode live --routers rule,jev \
    --sample jev_live_v1 --experiments main,order,distractors \
    --order-permutations 2 --distractor-levels 16,48                # B: benchmark
python -m experiments.routing.benchmark --mode live --routers jev \
    --sample jev_live_v1 --experiments main --repetitions 3         # C: consistency
python -m experiments.routing.benchmark --mode live --routers rule,jev \
    --split eval --kinds workflow --experiments main                # workflow completion
python -m experiments.routing.real_jev uav --out <dir>              # UAV branches A-D
python -m experiments.routing.real_jev publish --run smoke=<dir> ... --uav <dir>
pytest -m live_jev                                                  # opt-in live tests
```

`publish` writes a sanitized summary to `experiments/routing/results/published/`
(the only committed results folder). Results are labelled `REAL_JEV`, `RULE_ROUTER` or
`FAKE_JEV`, and never merged. The writer refuses any text that looks like a credential.

## 4. Verification performed

| Check | How | Live? |
|---|---|---|
| Request body and endpoint | real SDK clients with `httpx2.MockTransport`; bodies asserted | no |
| Strict vs production request shape | same | no |
| Served-model mismatch rejection | same | no |
| Usage and served-model recording | same | no |
| Timeout, 404, 429, 500, refusal, truncation | same | no |
| Malformed model text → routing failure | same | no |
| Preflight (`models.retrieve` / `models.list`) | same | no |
| Model discovery, bounded retries, 401/403/404/422/429/5xx classes, deadline, malformed answers, confidence parsing, redaction | `tests/integrations/test_typesafe_jev_reliability.py` (real SDK, mock transport) | no |
| UAV workflow under misbehaving routers (repeated step, prohibited step, step budget, malformed, timeout, early finish) | `tests/apps/test_uav_routing_reliability.py` | no |
| One real call per provider | `tests/integrations/test_live_smoke.py`, skipped unless `JEVPILOT_LIVE_TESTS=1` plus key | Jev: **passed (2026-09-29)**; Claude as a *router*: not run |
| Claude through LangChain (interpretation) + real Jev, end to end | `python -m experiments.routing.real_jev fully-live` | **yes: 5/5 runs completed (2026-09-29)**, section 8 |
| Real Jev: auth + discovery, one decision with probabilities, one UAV run | `tests/live/test_real_jev.py`, `pytest -m live_jev`, skipped without `TYPESAFE_API_KEY` | **yes: 3 passed (2026-09-29)** |

For Jev, the model identifier and the adapter were verified against the live API
(section 7). For Claude, the model identifier (`models.retrieve`) and the LangChain
interpretation path were verified live (section 8). Claude as a routing model
(`LLMRouter`) has not been run live.

## 5. UAV materials workflow with live services

| Mode | Interpretation | Routing | Needs |
|---|---|---|---|
| `OFFLINE_DEMO` (`--demo`) | deterministic parser, or a scripted chat model through LangChain | scripted policy via `FakeJevAdapter` | nothing |
| `LIVE_ROUTING` (`--live-routing`) | offline, as above: **not fully live**, and labelled so | real Jev | `TYPESAFE_API_KEY` |
| `LIVE` (`--live`) | a real LLM, natively or through LangChain (`--llm-backend`) | real Jev | `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` |

```bash
pip install -e '.[anthropic,jev,langchain]'
export TYPESAFE_API_KEY="..." ANTHROPIC_API_KEY="..."        # placeholders only
jevpilot uav-materials --live --llm-backend langchain --request "your request"
JEVPILOT_LIVE_TESTS=1 pytest tests/apps/test_uav_live.py -s     # opt-in end-to-end test
```

Missing keys or SDKs, or a failed Jev preflight (authentication, model), stop the
command before anything runs, with exit code 2 and a clear message. There is no
silent fallback to the offline demo. The workflow
capabilities take no inputs, so Jev only chooses *which* step runs next and
never has to produce a value (see "Unavoidable differences" above). The live
end-to-end run has **not** been performed in the development environment
(no credentials).

## 6. LangChain (interpretation only)

LangChain is an optional communication and structured-output layer for turning the
request into requirements. `LangChainRequirementInterpreter` implements the same
`RequirementInterpreter` contract as the native `LLMRequirementInterpreter`, sends the
same system and user prompt, requests the same JSON schema (`InterpretationPayload`)
via `with_structured_output` (tool calling), and validates the answer with the same
`parse_interpretation`. A failed call or a missing tool call is a visible failed
interpretation, never a fallback.

The provider comes from `JEVPILOT_LANGCHAIN_MODEL` (`provider:model`, default Anthropic
with the native adapter's model). Only `langchain-core` and `langchain-anthropic` are
used: no `langchain` meta-package and no LangGraph. The core, the domain, the Jev
adapter and the experiments never import LangChain (enforced by tests).

LangChain reduces provider-specific coupling. It does not make models
interchangeable: they still differ in schema adherence, latency, cost and
interpretation quality. **LangChain never routes**: Jev (or another JevPilot router)
chooses every step.

## 7. Live Jev validation (2026-09-29)

TypeSafe System One API through `TypeSafeJevAdapter`, with authentication OK. The model
was discovered as `jev-preview` (newest listed; `jev-latest` was also listed), and the API
reports it as served by `jev-1.13.0`. Requirements were interpreted offline (REAL_JEV,
not fully live). The cases were frozen before the runs
(`benchmarks/routing/samples/jev_live_v1.json`: 32 eval decision cases across levels
L1–L5), and neither fallback nor tuning was used. Sanitized results:
`experiments/routing/results/published/real_jev_2026-09-29.json`.

Stages: A, smoke (11 decisions); B, the 32 cases plus 2 capability-order permutations
and 16 / 48 offered capabilities; C, the 32 cases repeated 3 times; plus the 29 frozen
eval workflows run end to end.

| Measured (Stage B unless noted) | REAL_JEV | RULE_ROUTER |
|---|---|---|
| Valid typed decisions | 93.8% (30/32) | 100% (32/32) |
| Acceptable next action | 65.6% (21/32) | 68.8% (22/32) |
| Preferred action | 65.6% (21/32) | 40.6% (13/32) |
| Forbidden action chosen | 0/32 | 0/32 |
| Ask-human recall / precision | 3/7 / 3/3 | 0/7 / – |
| Finish recall (premature finishes) | 2/2 (0/23) | 2/2 (0/23) |
| Accuracy on paraphrased goals | 53.8% (7/13) | 30.8% (4/13) |
| Decision changed by capability order | 18.8% (6/32) | 0% |
| Accuracy with 16 / 48 capabilities offered | 68.8% / 75.0% | 71.9% / 71.9% |
| Workflow completion (29 eval workflows) | 89.7% (26/29) | 62.1% (18/29) |
| Unnecessary calls in those workflows | 28.3% (34/120) | 7.0% (4/57) |
| Routing latency p50 / p95 | 231 / 409 ms | < 1 ms |

Findings:

- **Repetition (Stage C).** The same decision in all 3 repetitions for 31/32 cases;
  accuracy per repetition was 65.6%, 65.6% and 68.8%.
- **Confidence.** Jev's self-report, with no calibration claimed: median 0.99 for
  acceptable decisions (n = 64) and 0.445 for the rest (n = 26), in Stage C.
- **Operations.** 540 benchmark API requests with 0 errors, 0 rate limits and 0 retries.
  A decision takes two calls when inputs are needed.
- **Missing information.** In the missed ask-human cases, Jev chose a step whose
  required input is absent. The grounded-input question then had no value to pick, and
  the decision was rejected as `InvalidRoutingInputError`. This is the designed
  behaviour (no value is guessed), and it accounts for every invalid decision.
- **Workflows.** Higher completion than the rule baseline, at the cost of more
  unnecessary calls.
- **UAV workflow (LIVE_ROUTING).** The existing-material, design and scripted-LangChain
  runs matched the reference path exactly. For the needs-information and
  unsupported-property requests, Jev asked for a human after the (correct) decision
  instead of generating the report. The application then reports "incomplete" with
  unchanged scientific content.

## 8. Fully live validation (2026-09-29)

The FULLY_LIVE path runs real Claude through LangChain, then real Jev, the controller,
the UAV capabilities, the scientific models and the report. Run it with
`python -m experiments.routing.real_jev fully-live`; sanitized results are in
`experiments/routing/results/published/fully_live_2026-09-29.json`. Claude was validated
as the **interpreter**, not as the router.

Claude `claude-opus-5` was both requested and served (repository default, confirmed
by `models.retrieve`). Jev `jev-preview` was served as `jev-1.13.0`. Expected decisions
are the frozen offline references.

| Frozen request | Claude's requirements | Decision (offline reference) | Report |
|---|---|---|---|
| Existing material (spar) | 4 hard, 3 soft | `use_existing` (same) | complete; design not run |
| Inverse design (built-in demo) | 4 hard, 4 soft: the offline parser's set | `design` (same) | complete |
| Needs information (no limits) | 0 hard, 3 soft | `needs_information` (same) | incomplete: Jev asked for a human before the report step |
| Unsupported property (temperature only) | 1 hard | `no_suitable_existing` (same) | incomplete: as above |
| CLI: `jevpilot uav-materials --live --llm-backend langchain` | – | `design` | complete, exit code 0 |

- **Runs.** 5 of 5 runs completed: 3 complete reports and 2 incomplete reports ending in
  a request for human input. There was also a Claude-only interpretation smoke test.
- **Interpretations.** The smoke test and the four branch runs were schema-valid, with
  no integrity findings and every limit quoted from the request.
- **Branch runs.** No untraced report numbers, no routing errors, no fallbacks, and
  design only after a `design` decision. The scientific results equal an offline replay
  of the same interpretation.
- **Latency.** Claude interpretation 4.0–10.4 s; Jev routing p50 113 ms and p95 308 ms
  (24 decisions); end to end 4.7–12.2 s per run.
- **Scale.** This is a small validation (5 runs) of one provider, not a benchmark of
  Claude.
