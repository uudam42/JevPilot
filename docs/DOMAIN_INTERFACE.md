# Adding a Domain

This guide shows how to add a new scientific, engineering or research domain
to JevPilot **without modifying any file under `jevpilot/`**.

A domain is a Python package that depends on `jevpilot` and subclasses
`DomainModule`. The two demo domains are full working references:
`domains/demo` (state subclass and reducer) and `domains/stats_demo` (artifacts,
uncertainty and provenance chains).

## 1. Checklist

| Provide | Required | Interface |
|---|---|---|
| Capabilities | yes | `Capability` (or `FunctionCapability`) with a `CapabilitySpec` |
| Evaluator(s) that define success | strongly recommended | `Evaluator` → `EvaluationResult` |
| State extension | optional | subclass `WorkflowState`, return it from `state_type()` |
| Reducers for extended fields | only if you extend state | `StateReducer` |
| Routing heuristics | optional | e.g. a list of `Rule`s for `RuleRouter` |
| Entry point | optional | `[project.entry-points."jevpilot.domains"]` |

## 2. Capabilities

```python
from pydantic import BaseModel
from jevpilot import (Capability, CapabilitySpec, CapabilityResult, ExecutionContext,
                      Artifact, ArtifactKind, SourceRef, StateEffects, Uncertainty)

class LookupInput(BaseModel):
    key: str

class Lookup(Capability):
    spec = CapabilitySpec(
        id="mydomain.lookup",               # "<domain>.<name>", globally unique
        name="lookup",
        description="Fetch a record by key from the reference table.",  # routers read this
        domain="mydomain",
        version="1.0.0",
        input_schema=LookupInput,
        tags=frozenset({"data", "read-only"}),
        preconditions=("reference table configured",),
        effects=("adds a record artifact",),
        cost_estimate=0.01, latency_estimate=0.2,
    )

    def is_applicable(self, state) -> bool:           # executable precondition
        return "table" in state.context

    def execute(self, inputs: LookupInput, ctx: ExecutionContext) -> CapabilityResult:
        record = ...                                   # your domain logic
        return CapabilityResult(
            output=record,
            artifacts=(Artifact(name=inputs.key, kind=ArtifactKind.JSON, content=record),),
            sources=(SourceRef(kind="database", identifier="ref_table", version="2024-01"),),
            uncertainty=Uncertainty(confidence=0.9),
            effects=StateEffects(context_updates={"last_key": inputs.key}),
        )
```

Rules for capabilities:

- **Read state only through `ctx.state`** (it is immutable), and propose changes
  by returning `StateEffects`, artifacts or candidates.
- **Signal failure by raising.** Use `CapabilityError(msg, retryable=False)` when
  retrying cannot help. Any other exception is treated as retryable.
- **Don't fill in `provenance` yourself.** The executor stamps it. Do report
  external `sources` and upstream `derived_from` provenance ids, so the lineage
  chain stays intact.
- **Declare Pydantic `input_schema` and `output_schema`.** They are validated by
  the executor and exported as JSON Schema to routers.

## 3. Evaluators: define "success"

```python
from jevpilot import Evaluator, EvaluationResult, ControlAction, ConstraintStatus

class MyEvaluator(Evaluator):
    evaluator_id = "mydomain.goal_met"
    def evaluate(self, state):
        ok = ...  # read goal.success_criteria, observations, artifacts
        return EvaluationResult(
            evaluator_id=self.evaluator_id, step=state.step,
            goal_progress=..., constraint_status=(ConstraintStatus(constraint_id="x", satisfied=ok),),
            remaining_gaps=() if ok else ("x not satisfied",),
            recommendation=ControlAction.TERMINATE_SUCCESS if ok else ControlAction.CONTINUE,
        )
```

Return `TERMINATE_FAILURE` for domain-defined infeasibility, `REPLAN` when the
approach should change, and `HUMAN_INTERVENTION` when judgement is needed. The
control policy applies generic budgets on top.

## 4. Extending the state (optional)

```python
class MyState(WorkflowState):
    best_score: float | None = None

class ScoreReducer(StateReducer):
    def reduce(self, state, observation):
        if isinstance(state, MyState) and observation.capability_id == "mydomain.score" \
                and observation.success:
            return state.evolve(best_score=max(state.best_score or 0, observation.result.score))
        return state      # ignore observations you don't own
```

Reducers must be pure and must ignore state types and observations that aren't
theirs. If the generic `context` / `StateEffects` path is enough, don't subclass
at all; `stats_demo` shows this style.

## 5. The domain module

```python
class MyDomain(DomainModule):
    name = "mydomain"
    version = "0.1.0"
    def capabilities(self): return [Lookup(), ...]
    def evaluators(self):   return [MyEvaluator()]
    def reducers(self):     return [ScoreReducer()]
    def state_type(self):   return MyState
```

`create_state(goal, **fields)` builds the initial state and tags
`metadata["domain"]`.

## 6. Running it

```python
from jevpilot import Runtime, RuleRouter, Goal

rt = Runtime()
dom = rt.load("mypackage.domain:MyDomain")   # or rt.load(MyDomain()) or rt.load("mydomain")
ctl = rt.controller(RuleRouter(my_rules), domains=["mydomain"])
result = ctl.run(dom.create_state(Goal(description="...", success_criteria={...})))
```

To make the domain discoverable by name, add this to your package's `pyproject.toml`:

```toml
[project.entry-points."jevpilot.domains"]
mydomain = "mypackage.domain:MyDomain"
```

## 7. Test your domain against the boundary

- Unit-test capabilities by calling `execute(inputs, ExecutionContext(...))` directly.
- Unit-test reducers and evaluators on hand-built states.
- Run your domain alongside `domains/demo` in one `Runtime` to check for id
  clashes and cross-domain interference.
- Your domain must not need any edit under `jevpilot/`. If it seems to, open a
  design discussion about the interface; don't special-case your domain in the core.
