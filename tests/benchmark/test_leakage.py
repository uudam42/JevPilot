"""Benchmark leakage guards: the answers must not leak into routers or baselines."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import experiments.routing.generalization.baselines as baselines
from experiments.routing.generalization import dataset as ds
from experiments.routing.generalization.world import load_catalog

ROOT = Path(__file__).resolve().parents[2]
CAT = load_catalog()


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_generic_routers_contain_no_benchmark_answers() -> None:
    ids = set(CAT.all)
    case_ids = {c.case_id for s in ds.SPLITS for c in ds.load_split(s)}
    facts = {f for d in CAT.all.values() for f in d.produces}
    for path in list((ROOT / "jevpilot").rglob("*.py")) + list(
        (ROOT / "integrations").rglob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        leaked = [x for x in ids | case_ids | facts if x in text]
        assert not leaked, f"{path} mentions benchmark items {sorted(leaked)[:5]}"


def test_baseline_rules_do_not_read_benchmark_data() -> None:
    path = Path(baselines.__file__)
    for name in _imports(path):
        assert not name.startswith(
            (
                "experiments.routing.generalization.dataset",
                "experiments.routing.generalization.oracle",
                "experiments.routing.generalization.runner",
            )
        ), name
    text = path.read_text(encoding="utf-8")
    for needle in ("read_text", "open(", "benchmarks/", "json.load", "eval_", "val_"):
        assert needle not in text, needle


def test_rule_keywords_all_come_from_dev_goals() -> None:
    dev_goals = " ".join(s.goal.lower() for s in ds.load_split("dev"))
    source = Path(baselines.__file__).read_text(encoding="utf-8")
    body = source.split("def _wants", 1)[1].split("return w", 1)[0]
    keywords = re.findall(r'"([a-z ]+)" in g', body)
    assert keywords
    assert [k for k in keywords if k not in dev_goals] == []


def test_integrations_and_routers_do_not_import_benchmarks() -> None:
    for path in list((ROOT / "integrations").rglob("*.py")) + list(
        (ROOT / "jevpilot").rglob("*.py")
    ):
        for name in _imports(path):
            assert not name.startswith(("experiments", "benchmarks", "tests")), (path, name)


def test_oracle_is_only_used_by_benchmark_code() -> None:
    allowed = {"dataset.py", "runner.py", "oracle.py"}
    for path in (ROOT / "experiments").rglob("*.py"):
        if any("generalization.oracle" in n for n in _imports(path)):
            assert path.name in allowed, path
    for path in (ROOT / "jevpilot").rglob("*.py"):
        assert "oracle" not in path.read_text(encoding="utf-8").lower(), path


def test_eval_goal_wording_is_not_in_the_rules() -> None:
    data = json.loads((ROOT / "benchmarks/routing/eval/workflows.json").read_text())
    paraphrased = [c for c in data["cases"] if "paraphrased_goal" in c["categories"]]
    assert paraphrased
    source = Path(baselines.__file__).read_text(encoding="utf-8").lower()
    for word in ("irregular", "lookup table", "write-up", "contrast", "responsible"):
        assert word not in source
