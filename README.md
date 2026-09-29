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
- Real Jev: the benchmark configuration is frozen (32 eval decision cases in
  `benchmarks/routing/samples/jev_live_v1.json`, plus order and distractor
  perturbations, 3 repetitions, and the four UAV branches). **No real-Jev result is
  reported yet**: the live runs have not been executed in this environment.

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
- Live LLM interpretation and live Jev routing depend on external services; see Testing
  for what has actually been measured.

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
