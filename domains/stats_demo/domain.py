"""Stats demo domain: estimate a process mean to a requested precision."""

from __future__ import annotations

from collections.abc import Sequence

from domains.stats_demo.capabilities import all_capabilities
from jevpilot import (
    ArtifactKind,
    Capability,
    ConstraintStatus,
    ControlAction,
    DomainModule,
    EvaluationResult,
    Evaluator,
    Goal,
    Rule,
    WorkflowState,
)


def _latest_dataset_id(state: WorkflowState) -> str | None:
    art = state.latest_artifact(ArtifactKind.DATASET)
    return art.id if art else None


def _summarized(state: WorkflowState) -> bool:
    ds = _latest_dataset_id(state)
    return ds is not None and state.context.get("summary_of") == ds


def _precise_enough(state: WorkflowState) -> bool:
    hw = state.context.get("ci_half_width")
    return hw is not None and hw <= state.goal.success_criteria["ci_half_width"]


def _reported(state: WorkflowState) -> bool:
    return _summarized(state) and state.context.get("report_for") == state.context["summary_of"]


class PrecisionEvaluator(Evaluator):
    """Success = a report exists for an estimate whose 95% CI is narrow enough."""

    evaluator_id = "stats.precision"

    def evaluate(self, state: WorkflowState) -> EvaluationResult:
        criteria = state.goal.success_criteria
        if "ci_half_width" not in criteria:
            return EvaluationResult(evaluator_id=self.evaluator_id, rationale="not a stats goal")
        precise = _summarized(state) and _precise_enough(state)
        constraint = ConstraintStatus(
            constraint_id="ci_half_width",
            satisfied=precise if _summarized(state) else None,
            detail=f"half width {state.context.get('ci_half_width')} "
            f"vs required {criteria['ci_half_width']}",
        )
        base = dict(
            evaluator_id=self.evaluator_id, step=state.step, constraint_status=(constraint,)
        )
        if precise and _reported(state):
            return EvaluationResult(
                **base,
                goal_progress=1.0,
                recommendation=ControlAction.TERMINATE_SUCCESS,
                rationale="precise estimate reported",
            )
        if state.context.get("sample_size", 0) > criteria.get("max_samples", float("inf")):
            return EvaluationResult(
                **base,
                recommendation=ControlAction.TERMINATE_FAILURE,
                rationale="sample budget exceeded before reaching precision",
            )
        gaps = []
        if not _summarized(state):
            gaps.append("latest data not summarized")
        elif not precise:
            gaps.append("interval too wide")
        else:
            gaps.append("report missing")
        return EvaluationResult(
            **base,
            goal_progress=0.8 if precise else 0.3,
            remaining_gaps=tuple(gaps),
            rationale="; ".join(gaps),
        )


class StatsDomain(DomainModule):
    name = "stats"
    version = "0.1.0"
    description = "Toy domain: sample a noisy process until the mean is known precisely."

    def capabilities(self) -> Sequence[Capability]:
        return all_capabilities()

    def evaluators(self) -> Sequence[Evaluator]:
        return [PrecisionEvaluator()]

    def new_workflow(
        self, true_mean: float, noise: float, ci_half_width: float, max_samples: int = 10_000
    ) -> WorkflowState:
        goal = Goal(
            description=f"estimate the process mean to ±{ci_half_width}",
            parameters={"true_mean": true_mean, "noise": noise},
            success_criteria={"ci_half_width": ci_half_width, "max_samples": max_samples},
        )
        return self.create_state(goal)


def routing_rules(initial_n: int = 8, growth: int = 4) -> list[Rule]:
    """Domain-supplied heuristics for the generic RuleRouter."""
    return [
        Rule(
            "stats.sample",
            when=lambda s: _latest_dataset_id(s) is None,
            inputs={"n": initial_n, "seed": 1},
            reason="no data yet",
        ),
        Rule("stats.summarize", when=lambda s: not _summarized(s), reason="summarize new data"),
        Rule(
            "stats.sample",
            when=lambda s: not _precise_enough(s),
            inputs=lambda s: {"n": s.context["sample_size"] * growth, "seed": s.step},
            reason="interval too wide; collect more data",
        ),
        Rule("stats.report", when=lambda s: not _reported(s), reason="precise; write report"),
    ]
