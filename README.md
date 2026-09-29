English | [中文](README.zh-CN.md)

# JevPilot

JevPilot is a domain-agnostic agentic orchestration framework combining an optional
LangChain-based LLM communication layer, TypeSafe Jev decision routing, specialized
executable capabilities, and deterministic scientific models.

**Status:** frozen v0.1 research prototype; no API stability promised.

## What JevPilot Does

It turns a plain-English engineering request into an evidence-backed report. Each part
does one job:

| Part | Job | Never does |
|---|---|---|
| Claude (via LangChain or natively) | interprets the request into validated, structured requirements | invent a material property, limit or unit |
| TypeSafe Jev | decides which step runs next (or to finish, or to ask a human) | execute anything |
| JevPilot | orchestration, shared state, validation, execution control | guess what the user meant |
| Capabilities | run the specialized workflow steps | route themselves |
| Scientific models | deterministic engineering calculation | claim more than their sources support |

This is agentic orchestration across specialized capabilities, not a swarm of autonomous
LLM agents. LangChain never routes.

## Architecture

```text
User
  ↓
Claude through LangChain / native LLM      natural language → structured requirements
  ↓
TypeSafe Jev                               decision routing
  ↓
JevPilot controller + shared state         validate · execute · update · evaluate · stop
  ↓
6 specialized executable capabilities
  ↓
UAV materials domain                       real data · requirements · evidence
  ↓
Scientific models                          micromechanics · lamination theory
  ↓
Engineering report
```

The core (`jevpilot/`) knows nothing about materials, LangChain or any provider. More
detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## UAV Materials Example

```text
request → structured requirements → search of 46 aerospace material records
        → evidence-aware decision (use existing / design / ask for information)
        → bounded composite inverse design if needed (28 designs)
        → micromechanics + Classical Lamination Theory → report
```

For the built-in wing-skin request (density ≤ 1800 kg/m³, stiffness ≥ 40 GPa in x and y,
shear stiffness ≥ 15 GPa), no existing material qualifies. Of the 28 designs searched, 8
are feasible, and the best is a virtual [0/45/-45/90]s carbon/epoxy laminate (V_f 0.65,
predicted 1579 kg/m³, Ex = Ey = 53.4 GPa, Gxy = 20.4 GPa). Its strength, corrosion and
temperature behaviour are reported as not predicted. Method:
[docs/UAV_MATERIALS.md](docs/UAV_MATERIALS.md).

## Quick Start

```bash
git clone https://github.com/uudam42/JevPilot.git && cd JevPilot
uv venv && uv pip install -e '.[dev]'        # or: python -m venv .venv && pip install -e '.[dev]'
jevpilot uav-materials --demo                 # offline and deterministic: no key, no network
jevpilot uav-materials --demo --request "The density must not exceed 3000 kg/m^3 ..."
jevpilot uav-materials --demo --output report.md --json report.json
```

Offline runs are labelled **OFFLINE_DEMO** and say nothing about model quality.

## Live Mode

```bash
uv pip install -e '.[jev,langchain]'
export TYPESAFE_API_KEY="..."                 # placeholders only: never commit a real key
export ANTHROPIC_API_KEY="..."
jevpilot uav-materials --live --llm-backend langchain --request "..."   # Claude + Jev
jevpilot uav-materials --live-routing          # real Jev routing, offline interpretation
pytest -m live_jev                             # opt-in live tests
```

Keys are read only from the environment. If credentials are missing, the command stops
with exit code 2; it never falls back to the demo. LangChain (optional:
`langchain-core` + `langchain-anthropic`, no LangGraph) reduces provider coupling, but
models still differ in schema adherence, latency and cost. Setup details:
[docs/REAL_ROUTING.md](docs/REAL_ROUTING.md).

## Validation Summary

Measured on 2026-09-29. Claude `claude-opus-5`; Jev `jev-preview`, served as `jev-1.13.0`.

- **Fully live** (real Claude → LangChain → real Jev → JevPilot → UAV workflow): 5/5 runs
  completed. 3 produced complete reports. In the other 2 (needs-information and
  unsupported-property), Jev asked for human input before the report step, so those
  reports are marked incomplete. All decisions matched the frozen offline references.
- **Real Jev benchmark:** 540 API requests, 0 API errors. The same decision on 31/32
  frozen cases across 3 repetitions. 89.7% completion (26/29) on the frozen workflow
  benchmark.
- **Latency (fully live):** Claude interpretation 4.0–10.4 s; Jev routing p50 113 ms and
  p95 308 ms (24 decisions); 4.7–12.2 s end to end.
- **Tests:** 843 passed, 3 skipped, 3 deselected (live) offline; `pytest -m live_jev`
  3 passed; 11/11 examples run.
- **Science:** outputs unchanged by the LangChain and Jev integration. Claude was
  validated as the interpreter, not as the router.

Full results, methodology and weaknesses found:
[docs/REAL_ROUTING.md](docs/REAL_ROUTING.md) (sections 7–8) and the sanitized results in
[`experiments/routing/results/published/`](experiments/routing/results/published/).

## Scientific Scope

- 46 aerospace material records (MIL-HDBK-5J, NRL and NASA reports), with traceable evidence
- NASA-sourced ply micromechanics and Classical Lamination Theory, each checked against its sources
- bounded composite inverse design: 4 fibre volume fractions × 7 symmetric layups
- missing data is an evidence gap; predictions are never presented as measurements

## Limitations

- A research prototype; the acceptance policy is a demonstration, not a design standard.
- Small dataset with gaps; composite strength, failure, corrosion, moisture and
  temperature are not predicted; the models are not experimentally validated.
- The live validation is small: one frozen benchmark and 5 fully live runs, with one LLM
  provider. It is evidence, not a general guarantee.

## Repository and Documentation

```text
jevpilot/       core: state, control loop, routers, validation
integrations/   TypeSafe Jev, Anthropic and LangChain adapters; redaction
domains/        uav_materials (science + workflow); two toy domains for tests
apps/           UAV materials application and CLI
benchmarks/, experiments/   frozen routing benchmark, harness, published results
tests/, examples/, docs/
```

[ARCHITECTURE](docs/ARCHITECTURE.md) · [UAV_MATERIALS](docs/UAV_MATERIALS.md) ·
[REAL_ROUTING](docs/REAL_ROUTING.md) · [BENCHMARK_DESIGN](docs/BENCHMARK_DESIGN.md)

No software license has been chosen yet. The material data come from U.S. Government
public documents (reuse basis and checksums in `data/uav_materials/sources/`). TypeSafe
Jev and Anthropic are third-party services under their own terms.
