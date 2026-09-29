English | [中文](README.zh-CN.md)

# JevPilot

JevPilot is a domain-agnostic agentic orchestration framework combining an optional
LangChain-based LLM communication layer, TypeSafe Jev decision routing, specialized
executable capabilities, and deterministic scientific models. It runs an explicit,
inspectable control loop over shared workflow state, and ships with a UAV materials
workflow that goes from a plain-English request to an evidence-aware engineering report.

**Status:** experimental research prototype (v0.1.0); no API stability promised.

## Why JevPilot

Engineering answers need numbers you can trust, so each part does only its own job:

| Role | Component | Never does |
|---|---|---|
| **Interprets** | LLM (native adapter or LangChain) | invent a material property, limit or unit |
| **Decides** | TypeSafe Jev chooses the next step | execute anything |
| **Orchestrates** | JevPilot controller + shared state | guess what the user meant |
| **Executes** | specialized capabilities (search, design, report) | route themselves |
| **Calculates** | scientific models and real data | claim more than their sources support |

This is agentic orchestration across specialized capabilities, not a swarm of
autonomous LLM agents: one router picks among capabilities whose preconditions hold,
and every decision is validated before anything runs.

## Architecture

```text
User request
  ↓
LangChain / native LLM interpreter      → validated, structured requirements
  ↓
TypeSafe Jev                            → next capability (or finish / ask a human)
  ↓
JevPilot controller + shared state      → execute · update · evaluate · stop
  ↓
Specialized capabilities
  ↓
Scientific domain (UAV materials)       → data search · micromechanics · CLT
  ↓
Evaluation → report
```

The core (`jevpilot/`) knows nothing about materials, LangChain or any provider. The
domain depends only on the core's public API. The application (`apps/`) chooses the
interpreter and the router.

## UAV Materials Demo

```text
natural-language request
 → structured requirements (every number must be quoted from the request)
 → search of 46 real aerospace materials, with provenance
 → evidence-aware decision: use an existing material, design, or ask for information
 → if needed, bounded composite inverse design (fibre fraction × symmetric layups)
 → micromechanics + Classical Lamination Theory predictions, same evaluator
 → structured report + Markdown, built only from computed state
```

For the built-in wing-skin request (density ≤ 1800 kg/m³, stiffness ≥ 40 GPa in x and
y, shear stiffness ≥ 15 GPa), no existing material qualifies, and the design branch
proposes a virtual [0/45/-45/90]s carbon/epoxy laminate (V_f 0.65, predicted 1579 kg/m³,
Ex = Ey = 53.4 GPa, Gxy = 20.4 GPa). Its strength, corrosion and temperature behaviour
are stated as not predicted.

## Quick Start

```bash
git clone https://github.com/uudam42/JevPilot.git && cd JevPilot
uv venv && uv pip install -e '.[dev]'        # or: python -m venv .venv && pip install -e '.[dev]'
jevpilot uav-materials --demo                 # offline, deterministic, no key, no network
jevpilot uav-materials --demo --request "The density must not exceed 3000 kg/m^3 ..."
jevpilot uav-materials --demo --output report.md --json report.json --routing-log routing.jsonl
```

`--demo` runs are labelled **OFFLINE_DEMO**. They use a deterministic phrase parser and a
scripted routing policy, and say nothing about model quality.

## LangChain Integration

Optional (`pip install -e '.[langchain]'`: `langchain-core` + `langchain-anthropic` only,
no LangGraph). It provides a model-provider abstraction and structured output, which
reduces coupling to one provider's SDK:

```bash
jevpilot uav-materials --demo --llm-backend langchain                 # offline scripted chat model
jevpilot uav-materials --live --llm-backend langchain --request "..." # real LLM + real Jev
```

`LangChainRequirementInterpreter` implements the same interpreter contract as the native
one, sends the same prompt, requests the same schema (tool-calling structured output),
and passes the answer through the same validation. **LangChain does not perform Jev
routing** and does not replace the controller. Different models can still differ in
schema adherence, latency, cost and interpretation quality.

## Real Jev Setup

```bash
uv pip install -e '.[jev]'
export TYPESAFE_API_KEY="..."                  # placeholder: never commit or paste a real key
python -m experiments.routing.real_jev preflight          # authentication + model discovery
jevpilot uav-materials --live-routing                     # real Jev routes; interpretation offline
pytest -m live_jev                                        # opt-in live tests
```

The key is read only from the environment. The model is discovered from `GET /v1/models`
(or set `TYPESAFE_DEFAULT_MODEL`), and routing calls go to `POST /v1/systemone` through the
official SDK. `--live` also needs `ANTHROPIC_API_KEY`. A live run with missing credentials
stops with exit code 2 and never falls back to the demo. Details:
[docs/REAL_ROUTING.md](docs/REAL_ROUTING.md).

## Testing

```bash
pytest                                                     # offline: no key, no network
ruff check . && ruff format --check .
mypy --strict jevpilot domains experiments integrations examples apps
```

- Offline suite: **842 passed, 3 skipped** (older opt-in live tests), 3 deselected (`live_jev`).
- Live tests: `pytest -m live_jev` **3 passed** against the real service.

**Real Jev** (TypeSafe System One API, 2026-09-29). The model was discovered from
`GET /v1/models` as `jev-preview`, and the API reports serving it as `jev-1.13.0`. The
cases were frozen before the runs (`benchmarks/routing/samples/jev_live_v1.json`, 32
eval decision cases across levels L1–L5), there was no fallback, and nothing was tuned
afterwards. Full sanitized results:
[`real_jev_2026-09-29.json`](experiments/routing/results/published/real_jev_2026-09-29.json).

| Measured | REAL_JEV | RULE_ROUTER |
|---|---|---|
| Valid typed decisions (32 cases) | 93.8% (30/32) | 100% (32/32) |
| Acceptable next action | 65.6% (21/32) | 68.8% (22/32) |
| Preferred action | 65.6% (21/32) | 40.6% (13/32) |
| Forbidden action chosen | 0/32 | 0/32 |
| Ask-human recall / precision | 3/7 / 3/3 | 0/7 / – |
| Finish recall (premature finishes) | 2/2 (0/23) | 2/2 (0/23) |
| Accuracy on paraphrased goals | 53.8% (7/13) | 30.8% (4/13) |
| Decision changed by capability order (2 permutations) | 18.8% (6/32) | 0% |
| Accuracy with 16 / 48 capabilities offered | 68.8% / 75.0% | 71.9% / 71.9% |
| Workflow completion (29 eval workflows) | 89.7% (26/29) | 62.1% (18/29) |
| Unnecessary calls in those workflows | 28.3% (34/120) | 7.0% (4/57) |
| Routing latency p50 / p95 (Stage B) | 231 / 409 ms | < 1 ms |

- Repetition (3 runs of the 32 cases): the same decision in 31/32 cases; accuracy per
  run 65.6%, 65.6%, 68.8%.
- Confidence (Jev's self-report, no calibration claimed): median 0.99 for acceptable
  decisions (n = 64) and 0.445 for the rest (n = 26).
- API: 540 benchmark requests, 0 errors, 0 rate limits, 0 retries.
- Invalid decisions all came from missing-information cases: Jev chose a step whose
  required input is absent instead of asking a human, and JevPilot rejected it rather
  than guessing a value.
- UAV workflow with real Jev routing (LIVE_ROUTING; requirements interpreted offline):
  the existing-material, design and LangChain-interpreter runs follow the reference
  path, give the same decision and scientific content, and produce complete reports
  (design never runs on the existing-material branch). For the needs-information and
  unsupported-property requests, Jev reaches the correct decision and then asks for a
  human instead of generating the report, so the report is marked incomplete. Its
  scientific content is identical.
- No live LLM run yet (no Anthropic key): the LangChain path was exercised with a
  scripted offline chat model and real Jev routing.

## Scientific Scope

- 46 real aerospace material records (MIL-HDBK-5J, NRL and NASA reports) with provenance
- NASA-sourced ply micromechanics (TM-83320, TP-3290), checked against the sources' examples
- Classical Lamination Theory (NASA RP-1351), checked against its printed matrices
- bounded composite inverse design over a small, explicit grid
- evidence handling: missing data is an evidence gap, predictions are never measurements

## Limitations

- Research prototype; the acceptance policy is a demonstration, not a design standard.
- Small dataset (46 records) with gaps; constituent data are handbook values.
- Composite strength, failure, corrosion, moisture and temperature are not predicted.
- Models are checked against their sources, not validated by experiments.
- Live LLM interpretation has not been run yet; real Jev routing was measured on a small
  frozen benchmark (see Testing), which is evidence, not a general guarantee.

## Repository Structure

```text
jevpilot/        core: state, control loop, routers, validation (no domain, no LangChain)
integrations/    Jev (TypeSafe SDK), Anthropic, LangChain chat models, redaction
domains/         uav_materials (science + workflow) and two toy domains for tests
apps/            uav_materials application: CLI, interpreters, offline/live wiring
benchmarks/, experiments/   routing benchmark data, harness and published live results
tests/, examples/, docs/
```

## License / Data Sources

No software license has been chosen yet; that decision is left to the project owner.
All material data come from U.S. Government public documents (MIL-HDBK-5J with
Distribution Statement A; NASA and NRL reports marked for public release). Source
manifests in `data/uav_materials/sources/` record the reuse basis, retrieval date and
checksums; source PDFs are not committed. TypeSafe Jev and Anthropic are third-party
services under their own terms.

More: [ARCHITECTURE](docs/ARCHITECTURE.md) · [UAV_MATERIALS](docs/UAV_MATERIALS.md) ·
[REAL_ROUTING](docs/REAL_ROUTING.md) · [BENCHMARK_DESIGN](docs/BENCHMARK_DESIGN.md)
