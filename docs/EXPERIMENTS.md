# Running Routing Experiments

## 1. Offline vs live

| Mode | Rule | LLM | Jev | Network | Purpose |
|---|---|---|---|---|---|
| `offline` | `RuleRouter` | `LLMRouter` + **FakeLLMAdapter** | `JevRouter` + **FakeJevAdapter** | none | infrastructure and regression tests |
| `live` | `RuleRouter` | `LLMRouter` + `AnthropicLLMAdapter` | `JevRouter` + `TypeSafeJevAdapter` | yes, costs money | model evaluation |

> **Offline results are infrastructure tests, not model-performance
> evidence.** The fake adapters run a naive word-overlap heuristic
> (`naive_request_policy`). Their numbers say nothing about Jev or any LLM.
> Manifests label offline runs `OFFLINE INFRASTRUCTURE TEST`.

In live mode:

- a router whose SDK, credentials or model preflight is missing is marked
  **unavailable** in the manifest and is **not run**. Nothing is substituted
  for it;
- fake adapters are rejected, even inside a fallback chain;
- strict single-model mode is the default (`--no-strict` is recorded).

## 2. Commands

```bash
# offline, main experiment, eval split
python -m experiments.routing.benchmark --mode offline --routers rule,llm,jev --split eval

# everything, offline
python -m experiments.routing.benchmark --mode offline --split eval \
    --experiments main,order,names,distractors,fallback --repetitions 3

# LIVE: the Phase 1.5 deliverable (needs ANTHROPIC_API_KEY and/or TYPESAFE_API_KEY)
python -m experiments.routing.benchmark --mode live --routers rule,llm,jev \
    --split eval --repetitions 5 --strict \
    --llm-model claude-opus-5 --llm-price 5,25 --pricing-source "<where/when>"

# LIVE validation before any full run: one smoke call per provider, then an
# 11-case engineering sample (benchmarks/routing/samples/tiny_live_v1.json)
JEVPILOT_LIVE_TESTS=1 pytest tests/integrations/test_live_smoke.py -s
python -m experiments.routing.benchmark --mode live --routers rule,llm,jev \
    --sample tiny_live_v1 --repetitions 1 --strict

# develop against dev and validation, not eval
python -m experiments.routing.benchmark --mode live --routers llm --split validation
```

Useful flags:

| Flag | Effect |
|---|---|
| `--experiments` | any of `main,order,names,distractors,fallback` |
| `--kinds` | `decision`, `workflow` or both |
| `--order-permutations` | default 3 |
| `--distractor-levels` | default 8,16,32,48 |
| `--systems` | fallback chains, e.g. `jev>rule,llm>rule` |
| `--faults` | offline only: inject faults into fakes |
| `--seed` | benchmark seed |
| `--timeout` | per routing call, seconds |
| `--no-traces` | skip per-workflow JSONL traces |

The Phase 1 smoke benchmark (pipeline, arithmetic and stats suites) is
`python -m experiments.routing.smoke`.

## 3. Cost of a live run (rough)

Measured prompt size on eval: about 22.8k characters (about 5–7k tokens)
per decision, rising to about 38k characters at 48 capabilities. The main
experiment makes about 101 decision calls plus about 115 workflow calls
(more if routers take extra steps). With Claude's adaptive thinking adding
output tokens, expect roughly $10–20 per repetition of `main` for an
Opus-tier model at published list prices ($5 input / $25 output per MTok as
of mid-2026; **verify current pricing**). The `order`, `names` and
`distractors` experiments multiply this by about 3, 1 and 4. Jev uses 1–2
calls per decision; its price was not available here. Start with
`--split validation --repetitions 1` and read `summary.json → usage`.

## 4. What a run writes

`experiments/routing/results/<run_id>/` (git-ignored):

| File | Answers |
|---|---|
| `manifest.json` | which router, adapter, provider, requested model and preflight metadata; prompt version and Jev request format; benchmark version, split, split and catalog digests, case ids; seeds and derivation; repetitions; strict flag; fallback systems; git commit and dirty flag; SDK, Python and platform versions; unavailable routers and why |
| `decisions.jsonl` | per decision: case, level, categories, unseen flag, perturbation, repetition, state facts, **request fingerprint**, offered count, decision (shown and canonical id), intent, inputs, validity or error, acceptable, preferred, forbidden, unnecessary, oracle sets, confidence, latency, attempts, providers, requested and actual models, tokens, cost |
| `requests.jsonl` | the exact `RoutingRequest` for every fingerprint: *what state and capabilities the router saw* |
| `workflows.jsonl` | per run: status vs expected, completed, forbidden reached, steps, reference steps, excess, calls, unnecessary and failed calls, router retries, final intent, capability sequence, routing failures and fallbacks, routing vs execution latency, tokens, trace file |
| `traces/<router>/*.jsonl` | the full controller trace of each workflow run, including every routing attempt with its request |
| `summary.json` / `summary.csv` | all metrics per router (per repetition, then aggregated), slices, robustness |

## 5. Reproducibility

- Offline runs are deterministic given the seed, apart from latencies. A test
  checks this.
- Live model outputs are **not** assumed deterministic. Use
  `--repetitions N`. Every record carries its repetition, and metrics report
  the spread across repetitions.
- Reproducing a run means re-running with the manifest's commit, split
  digest, seeds, models, prompt version and flags. Remote model versions can
  still change behind an alias, so pin a model id when the provider allows it.

## 6. Reading results

Report **measured** numbers separately from **interpretation**:

1. Check `manifest.json`: `mode`, `routers.*.status`, `strict_single_model`
   and `actual_models`.
2. Compare routers only within one run (same split, seeds and commit).
3. Read RA with its denominator and its std across repetitions. On eval, one
   decision is about 1 percentage point, and one workflow is about 3.4
   points.
4. Read TCR together with UCR and excess steps. A router can complete by
   brute force.
5. For generalization, compare `unseen:yes` vs `unseen:no`, and
   `paraphrased_goal` vs `literal_goal`.
6. Finish and ask-human rates rest on few cases. Say so.
7. Do not claim one router is better than another unless the gap exceeds the
   repetition spread and holds across slices.
