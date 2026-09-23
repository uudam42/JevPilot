# Routing Generalization Benchmark: Design

The question this benchmark asks:

> Can a router control a stateful workflow? That means reacting to
> observations, avoiding unnecessary actions, recovering from failure,
> stopping at the right time, and composing known capabilities into
> workflows it has not seen.

It is **not** a tool-classification test. Every item is
$R(S_t, C_t) \rightarrow D_t$ with a full workflow state. Workflows run on
the unmodified `Controller` until the state satisfies the goal or the run
stops.

Data: `benchmarks/routing/`. Code: `experiments/routing/generalization/`.
Running it: [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. The world

An artificial, domain-neutral **analysis toolkit** (no scientific content):

- **State** is a set of facts (`context.facts`, for example `dataset_loaded`,
  `records_cleaned`, `report_generated`).
- **22 core capabilities** (`catalog.json`). Each has a natural-language
  description, required facts (shown as preconditions), added facts (shown as
  effects), a JSON input schema and a cost. Some inputs must equal a goal
  parameter, for example `source`.
- **36 distractors** (`distractors.json`). They are plausible but irrelevant,
  and some sound close to real steps (`pack_archive` vs `archive_results`,
  `export_contacts` vs `export_table`).
- **Traps by design:**
  - `quick_summary` produces a preview, not statistics;
  - `generate_report` vs `generate_illustrated_report`;
  - `drop_invalid_rows` is cheaper than `clean_records` but only after schema
    validation;
  - `load_dataset_mirror` is an alternative with higher cost;
  - `publish_public` is irreversible and forbidden in "internal only" goals.
- **Execution is honest.** Unmet preconditions, wrong input values and
  injected failures (transient or permanent) produce *failed observations*,
  not routing errors. The router must read them.

Routers see the goal **description** and parameters, the facts, the action
history with results and errors, and the capability descriptions. They do
**not** see the goal's target facts. A benchmark-side evaluator holds those.

## 2. Ground truth: an oracle, not a script

`oracle.py` computes the shortest number of successful actions from any
state to the goal, by exhaustive search that never produces a forbidden
fact. At each state:

| Kind | Meaning | Acceptable | Preferred |
|---|---|---|---|
| `invoke` | progress is possible | every offered, applicable capability on **some** shortest plan (with goal-bound inputs correct) | the cheapest of those |
| `finish` | the goal already holds | `finish`, `idle` | `finish` |
| `ask_human` | reachable only with missing information (a goal parameter) | `ask_human`, `idle`, or progress that doesn't need the missing value | `ask_human` |
| `impossible` | no offered capability can ever reach it | `ask_human`, `idle` | `ask_human` |

Also computed per state: *forbidden* (offered capabilities that would create
a forbidden fact) and *unnecessary* (everything else offered).

Consequences:

- `A → B` and `B → A` are **both correct** whenever both are optimal.
- A **workflow succeeds** because its final state contains the goal facts and
  no forbidden fact, never because it followed a particular sequence.
- **Excess steps** = steps − shortest plan length. That is a true lower bound,
  so the reference is defensible. It is only reported for completed runs
  that expected success.
- Transient failures are assumed retryable. A capability becomes unusable
  only after the router has **observed** it fail permanently.

## 3. Difficulty levels and categories

| Level | What it tests | Example |
|---|---|---|
| 1 direct | one obvious action, schema compliance | notify the owner by chat |
| 2 state-dependent | same goal and capabilities, different state → different action | "flag anomalous records" from fresh / loaded / schema-validated / cleaned states |
| 3 multi-step | 3–7 actions, trajectory-level evaluation | load → clean → statistics → plot → illustrated report |
| 4 failure recovery | react to a failed observation: retry or switch | transient statistics failure; primary store permanently down → mirror |
| 5 unseen composition | known parts, new combination | trend + outliers; compare groups + export |

Cross-cutting categories (tagged per case): `state_dependent`,
`multiple_valid`, `failure_recovery`, `missing_information`, `finish`,
`impossible`, `forbidden_action`, `distractors`, `similar_capabilities`,
`paraphrased_goal`, `literal_goal`.

## 4. Splits and what "unseen" means

| Split | Workflows | Decision cases | Use |
|---|---|---|---|
| `dev` | 23 | 71 | writing baselines and prompts |
| `validation` | 9 | 31 | checking changes before touching eval |
| `eval` | 29 | 101 | reporting only |

An eval case tagged `unseen_composition` must satisfy all three conditions,
checked by `dataset.unseen_violations()` in the test suite:

1. every capability on its reference plan is used on some **dev** reference plan (the parts are known);
2. its goal fact set occurs in **no dev or validation** case (the task is new);
3. at least one ordered transition `A → B` on its reference plan occurs on **no dev or validation** reference plan (the composition is new).

The test caught three cases that were initially mislabelled as unseen. They
were rewritten, not re-labelled.

**Wording shift.** Twelve eval goals are paraphrased with wording absent from
dev ("mark irregular records", "site-code lookup table", "write-up"). Three
unseen compositions also appear with dev-style wording
(`eval_u_*_literal`). That separates *composition* generalization from
*language* generalization, which were otherwise confounded.

## 5. Decision cases

Decision cases are **recorded, not written by hand**. The controller runs
each workflow with a recording oracle router along one reference trajectory,
and every state the controller presents becomes a case. States after failed
observations, including the real error messages, are therefore exactly
what a live router would see. Missing-information runs first make the
progress that needs no missing value, then ask. That yields "ask now" states.

Eval decision cases by expected kind: `invoke` 88, `ask_human` 8,
`impossible` 3, `finish` 2. Eval workflows by level: L1 2, L2 6, L3 7, L4 2,
L5 12 (14 tagged unseen, including the 2 L4 failure cases). The finish and ask-human samples are small.
Report their counts alongside their rates.

## 6. Perturbation experiments (decision level unless noted)

| Experiment | Change | Metric |
|---|---|---|
| `order` | K seeded permutations of capability order (default 3) | share of cases whose decision changes; accuracy per permutation |
| `names` | ids and names → opaque `cap_NN` (also in the action history). Descriptions, schemas, preconditions and effects unchanged. Catalog descriptions are tested to never mention capability names. | decision-change rate; accuracy (opaque) − accuracy (original) |
| `distractors` | N ∈ {8, 16, 32, 48} offered capabilities: all goal-relevant ones plus seeded irrelevant filler (decision **and** workflow) | accuracy, validity, unnecessary selections, latency, TCR, UCR per N |
| `fallback` | workflows run by fallback *systems* (`jev>rule`, `llm>rule`, `jev>llm>rule`) | reported separately as systems, never as a router's own score |

## 7. Metrics

**Decision level:**

- VDR: valid decisions / decisions.
- RA: acceptable decisions / decisions.
- Preferred rate.
- Invalid-capability, invalid-input and malformed rates (per router attempt).
- Forbidden and unnecessary selection rates.
- Ask-human recall and precision; *appropriate stop*, meaning `ask_human`
  or `idle` where stopping is correct.
- Finish recall (also explicit-finish only) and premature finish.
- Routing latency.
- Confidence when right vs wrong. Stored for calibration work, never
  treated as truth.
- Usage.

**Workflow level:**

- TCR: runs whose final status equals the expected status and that avoided
  forbidden facts / runs.
- Steps: mean, median, std, histogram.
- Excess steps.
- **UCR** = executed calls that were not on any shortest remaining plan when
  made / all executed calls. It is judged by replaying the oracle at every
  pre-call state.
- Forbidden-outcome, retry, replan, ask-human, routing-failure and fallback
  rates.
- Routing latency per decision, **separately** from execution latency per call.
- Total latency and usage.

**Robustness:** order sensitivity, name sensitivity, distractor curves,
failure-recovery success (TCR on `failure_recovery` cases) and
unseen-composition success (TCR on unseen cases). Slices by level, category
and unseen status are included.

Every rate carries its numerator and denominator. Each metric is computed per
repetition and reported as mean, median, std, min and max, plus the pooled
value.

## 8. Leakage protection

Enforced by tests (`tests/benchmark/test_leakage.py`, `test_generalization.py`):

- `jevpilot/` and `integrations/` contain no catalog ids, fact names or case ids.
- The rule baseline imports no dataset, oracle or runner code and reads no files.
- **Every keyword** in the rule baseline occurs in a **dev** goal, and none of
  the paraphrase vocabulary appears in it.
- The oracle is imported only by benchmark code.
- Live mode rejects fake adapters, even inside a fallback chain.
- The rule baseline solves 100% of dev and is tested **not** to solve all of
  eval.

The offline fake policy (`naive_request_policy`) is a generic token-overlap
heuristic over the request. It is not tuned on any split.

## 9. Known limitations

- **It is artificial.** One small world, one vocabulary, 22 capabilities.
  Results say how routers handle this structure, not how they handle real
  scientific workflows.
- **Lexical solvability.** A naive token-overlap heuristic completes about 69%
  of eval workflows (offline). Part of the benchmark is solvable without
  understanding, so read model results against that floor.
- **Small samples** for finish (2), impossible (3) and ask-human (8) decisions,
  only 2 L4 workflows, and 2–14 workflows per category. Differences of a few cases are noise.
- **"Unseen" is relative to reference plans.** A model's pre-training exposure
  to similar pipelines is unknowable.
- **Model inputs are grounded in goal parameters.** No task requires
  synthesising a new value, which favours selection-only routers such as Jev.
- **The oracle ignores cost when defining acceptable actions** (it uses step
  count; cost only decides *preferred*).
