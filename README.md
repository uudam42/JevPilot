# JevPilot

> A domain-agnostic agentic orchestration framework for scientific, engineering, and research workflows.

JevPilot runs an explicit, inspectable control loop over **structured state** and a
registry of **capabilities** (tools, models, simulators, databases, agents). The
policy that decides what to do next (a *router*) is pluggable, so the same
orchestration core can be driven by rules, an LLM, a learned policy, or Jev, and
can serve any domain that implements the domain interface.

```text
Goal → State → Plan/Route → Select Capability → Execute → Observe → Update State → Evaluate → Continue/Replan/Stop ↺
```

$$S_t \xrightarrow{\pi} A_t \xrightarrow{\text{execute}} O_t \rightarrow S_{t+1}$$

Domain applications such as materials discovery, robotics, battery design,
manufacturing or ML research are *examples* of what can be plugged in. None
of them is part of the core.

## Status

**Phase 1: routing intelligence, baselines and evaluation.** On top of the
Phase 0 core (loop, interfaces, registries, provenance, tracing, two trivial
demo domains), JevPilot now has a replaceable routing layer: `RuleRouter`,
`LLMRouter`, `JevRouter` and an explicit `FallbackRouter`. All of them consume
the same state and capability descriptions and return the same validated
`RoutingDecision`. A reproducible benchmark harness compares them.

| Integration | Status |
|---|---|
| Rule baseline | implemented, deterministic |
| LLM routing | `LLMRouter` + real `AnthropicLLMAdapter` (optional extra) + offline `FakeLLMAdapter` |
| Jev routing | `JevRouter` + offline `FakeJevAdapter`. **No real Jev adapter yet**; the `RoutingModelAdapter` contract is in place |
| Real-domain science | none, deliberately |

The simulated routers used by default in tests and benchmarks measure the
machinery, not model quality. No claim about the relative accuracy of Jev,
LLM or rule routing is made.

## Quick start

```bash
uv venv && uv pip install -e '.[dev]'     # or: pip install -e '.[dev]'
pytest                                   # unit, integration, architecture tests
python examples/minimal_workflow.py      # one domain, printed trace
python examples/cross_domain.py          # two unrelated domains, one core
python examples/compare_routers.py       # same workflow: rule vs LLM vs Jev vs fallback
python -m experiments.routing.benchmark  # routing benchmark (offline, simulated routers)
```

Everything above runs offline. For real Claude routing, install
`pip install -e '.[anthropic]'` and set credentials (see `.env.example`):

```python
from integrations.anthropic_llm import AnthropicLLMAdapter
from jevpilot import FallbackRouter, LLMRouter, RuleRouter
router = FallbackRouter(LLMRouter(AnthropicLLMAdapter(), timeout_s=60), RuleRouter(rules))
```

```python
from jevpilot import RuleRouter, Runtime
from domains.demo import ArithmeticDomain, routing_rules

runtime = Runtime()
domain = runtime.load(ArithmeticDomain())            # or runtime.load("domains.demo:ArithmeticDomain")
controller = runtime.controller(RuleRouter(routing_rules()))
result = controller.run(domain.new_workflow(start=3, target=20))

result.state.status      # WorkflowStatus.SUCCEEDED
result.state.provenance  # every observation / artifact traceable
result.trace             # machine-readable, step-by-step trace
```

## Layout

```text
jevpilot/
  core/           immutable data model: state, observation, artifact, provenance,
                  uncertainty, evaluation, decisions, plan, trace
  interfaces/     Capability, Router, Planner, Evaluator, ControlPolicy, StateReducer, DomainModule
  registry/       CapabilityRegistry, DomainRegistry (object / "module:Attr" / entry point)
  orchestration/  Controller (the loop), Executor, StateManager, DefaultControlPolicy, Runtime
  routing/        RoutingRequest, RuleRouter, ScriptedRouter, LLMRouter, JevRouter,
                  FallbackRouter, prompts, strict decision parsing, adapter contracts
  adapters/       offline fake LLM/Jev adapters with scripted fault injection
  planning/       reference planners: NullPlanner, StaticPlanner
domains/          example domains (depend on jevpilot; jevpilot never imports them)
  demo/           arithmetic toy domain (extends WorkflowState, uses a reducer)
  stats_demo/     sampling toy domain (artifacts, uncertainty, provenance chains)
integrations/     SDK-backed provider adapters (optional; jevpilot never imports them)
experiments/
  routing/        benchmark harness: fixtures, suites, metrics, CLI, results/
examples/  tests/{unit,integration,architecture,routing,benchmark}/  docs/
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): layers, control loop, abstractions, dependency rules
- [DESIGN_PRINCIPLES.md](docs/DESIGN_PRINCIPLES.md): the ten principles and how each is enforced
- [DOMAIN_INTERFACE.md](docs/DOMAIN_INTERFACE.md): adding a new domain without touching the core
- [STATE_MODEL.md](docs/STATE_MODEL.md): WorkflowState, Observation, Artifact, Provenance, EvaluationResult
- [ROUTING.md](docs/ROUTING.md): routing contract, RoutingRequest, LLM/Jev/fallback routers, validation, adding a router
- [BENCHMARKING.md](docs/BENCHMARKING.md): fixture format, metrics, reproducibility, running and reading benchmarks

## Requirements

Python ≥ 3.11. The only runtime dependency is `pydantic>=2.7`. Provider SDKs
are optional extras (`[anthropic]`).
