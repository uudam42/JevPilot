# Real Routing Integrations

JevPilot routes through adapters. The core never imports a provider SDK, and
real integrations live in `integrations/` as optional extras. This page
records what is real, how each integration was verified, and what is still
missing.

| Router | Adapter | Status (0.1.0) |
|---|---|---|
| `RuleRouter` | none | real, deterministic |
| `LLMRouter` | `integrations/anthropic_llm.py` (`AnthropicLLMAdapter`) | real code against `anthropic` 1.8.0; **verified through the SDK with a mock transport; no live call made yet** |
| `JevRouter` | `integrations/typesafe_jev.py` (`TypeSafeJevAdapter`) | real code against `typesafe-sdk` 0.7.1; **verified through the SDK with a mock transport; no live call made yet**; see [identity caveat](#is-this-the-intended-jev) |
| `LLMRouter` / `JevRouter` | `jevpilot/adapters` fakes | mocked, offline, infrastructure tests only |

Install: `pip install -e '.[anthropic]'` and/or `pip install -e '.[jev]'`.
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

### Findings (inspected 2026-09-23)

No Jev implementation, package or credential existed in this environment.
A public Jev was found and inspected from its official artefacts, and only
from those:

| Item | Finding (source: `typesafe-sdk` 0.7.1 source code and PyPI docs) |
|---|---|
| What it is | **Jev**, TypeSafe's "System One" model ([typesafe.ai](https://typesafe.ai), SDK repo `typesafe-ai/typesafe-sdk-python`) |
| Invocation | `TypeSafeClient(...).system_one(state, questions, model=..., timeout=...)`, which is `POST https://api.typesafe.ai/v1/systemone` |
| Input | `state`: text or JSON. `questions`: named `Choice` (≤ 255 labels with descriptions), `Noul` (yes/no), `Score` (ordered rubric) |
| Output | per question: `Choice` → `choice`, `confidence`, full `probabilities`; `Noul` → probability of yes; `Score` → expected score, confidence. Plus `model` (served) and `usage` (input/output tokens; output documented as free) |
| Models | default `jev-latest`; `client.models.list()` returns name, description, release date |
| Auth | `TYPESAFE_API_KEY` (also `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`) |
| Determinism controls | none exposed |
| Errors | typed; `TypeSafeAPITimeoutError` subclasses `TimeoutError`; HTTP status on `.status` |
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

### What is still missing for a live Jev run

1. Confirmation that TypeSafe Jev is the intended model.
2. A `TYPESAFE_API_KEY` for an account with access to the chosen model.
3. Optionally, the input-token price and its date, for cost estimates.

Then:

```bash
pip install -e '.[jev]'
JEVPILOT_LIVE_TESTS=1 pytest tests/integrations/test_live_smoke.py -k jev   # one call
python -m experiments.routing.benchmark --mode live --routers jev --split validation
```

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
| One real call per provider | `tests/integrations/test_live_smoke.py`, skipped unless `JEVPILOT_LIVE_TESTS=1` plus key | **not run: no credentials here** |

"Real model identifier is valid" and "adapter works against the live API"
therefore remain **unverified** until the smoke tests are run with
credentials.

## 5. UAV materials workflow in LIVE mode

The end-to-end application uses both integrations: Claude interprets the
request into structured requirements (validated like the offline parser's
output; no material values are accepted from the model), and Jev routes the
workflow capabilities.

```bash
pip install -e '.[anthropic,jev]'
export ANTHROPIC_API_KEY=... TYPESAFE_API_KEY=...
jevpilot uav-materials --live --request "your request"
JEVPILOT_LIVE_TESTS=1 pytest tests/apps/test_uav_live.py -s     # opt-in end-to-end test
```

Missing keys or SDKs stop the command with exit code 2 and a list of what is
missing; there is no silent fallback to the offline demo. The workflow
capabilities take no inputs, so Jev only chooses *which* step runs next and
never has to produce a value (see "Unavoidable differences" above). The live
end-to-end run has **not** been performed in the development environment
(no credentials).

