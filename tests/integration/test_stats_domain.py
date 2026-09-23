from domains.stats_demo import StatsDomain, routing_rules
from jevpilot import ArtifactKind, RuleRouter, Runtime, WorkflowStatus


def _run(**kw):  # type: ignore[no-untyped-def]
    rt = Runtime()
    domain = rt.load(StatsDomain())
    assert isinstance(domain, StatsDomain)
    return rt.controller(RuleRouter(routing_rules())).run(domain.new_workflow(**kw))


def test_reaches_precision_with_traceable_report() -> None:
    result = _run(true_mean=10.0, noise=3.0, ci_half_width=0.5)
    s = result.state
    assert s.status is WorkflowStatus.SUCCEEDED
    assert s.context["ci_half_width"] <= 0.5
    assert "mean" in s.uncertainty and s.uncertainty["mean"].interval

    # Provenance chain: report → candidate/summary → dataset → generator source
    by_id = {p.id: p for p in s.provenance}
    report = s.latest_artifact(ArtifactKind.REPORT)
    assert report and report.provenance
    summary_prov = by_id[report.provenance.derived_from[0]]
    assert summary_prov.capability_id == "stats.summarize"
    dataset_prov = by_id[summary_prov.derived_from[0]]
    assert dataset_prov.capability_id == "stats.sample"
    assert dataset_prov.sources[0].kind == "generator"


def test_domain_defined_failure() -> None:
    result = _run(true_mean=0.0, noise=100.0, ci_half_width=0.01, max_samples=100)
    assert result.state.status is WorkflowStatus.FAILED
    assert "sample budget" in result.control.reason
