"""Guards the core rule: DOMAIN → CORE is allowed, CORE → DOMAIN is forbidden."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "jevpilot"
CORE_FILES = sorted(CORE.rglob("*.py"))

# Allowed intra-package dependencies (layering). A subpackage may import itself too.
LAYERS: dict[str, set[str]] = {
    "core": set(),
    "exceptions": set(),
    "interfaces": {"core", "exceptions"},
    "registry": {"core", "interfaces", "exceptions"},
    "orchestration": {"core", "interfaces", "registry", "exceptions"},
    "routing": {"core", "interfaces", "exceptions"},
    "planning": {"core", "interfaces"},
    # Provider adapters depend on routing contracts; routing never imports adapters.
    "adapters": {"core", "interfaces", "routing", "exceptions"},
}

FORBIDDEN_THIRD_PARTY = {
    "openai",
    "anthropic",
    "langchain",
    "llama_index",
    "transformers",
    "torch",
    "requests",
}

# Domain vocabulary that must never appear in core code, comments or docstrings.
DOMAIN_WORDS = [
    "uav",
    "aerospace",
    "aircraft",
    "corrosion",
    "materials?",
    "alloy",
    "tensile",
    "thermal",
    "robot(ics)?",
    "battery",
    "batteries",
    "biology",
    "manufacturing",
    "fea",
    "composite mechanics",
]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _subpackage(path: Path) -> str:
    rel = path.relative_to(CORE).parts
    return rel[0].removesuffix(".py") if len(rel) > 1 or rel[0] != "__init__.py" else "__init__"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_core_never_imports_domains(path: Path) -> None:
    for name in _imports(path):
        assert name.split(".")[0] != "domains", f"{path} imports {name}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_layering(path: Path) -> None:
    layer = _subpackage(path)
    if layer == "__init__":
        return  # the top-level package re-exports everything by design
    allowed = LAYERS[layer] | {layer}
    for name in _imports(path):
        parts = name.split(".")
        if parts[0] == "jevpilot" and len(parts) > 1:
            assert parts[1] in allowed, f"{layer} must not import jevpilot.{parts[1]} ({path})"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_llm_or_heavy_framework_dependency(path: Path) -> None:
    for name in _imports(path):
        assert name.split(".")[0] not in FORBIDDEN_THIRD_PARTY, f"{path} imports {name}"


def test_core_has_no_domain_vocabulary() -> None:
    pattern = re.compile(r"\b(" + "|".join(DOMAIN_WORDS) + r")\b", re.IGNORECASE)
    hits = [
        f"{p.relative_to(ROOT)}:{i}: {line.strip()}"
        for p in CORE_FILES
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, "domain-specific vocabulary in core:\n" + "\n".join(hits)


def test_core_does_not_name_demo_domains() -> None:
    """The core must not special-case any loaded domain, not even the demos."""
    needles = ("arith.", "stats.", "ArithmeticDomain", "StatsDomain", "NumberState")
    for p in CORE_FILES:
        text = p.read_text(encoding="utf-8")
        for n in needles:
            assert n not in text, f"{p} mentions demo domain symbol {n!r}"


# Top-level packages that sit *outside* the core and may depend on it, never the reverse.
OUTSIDE_CORE = {"domains", "integrations", "experiments", "examples", "tests"}


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_core_never_imports_outside_packages(path: Path) -> None:
    for name in _imports(path):
        assert name.split(".")[0] not in OUTSIDE_CORE, f"{path} imports {name}"


# The generic loop must not know which routing policies or providers exist.
ROUTER_AGNOSTIC = ("core", "interfaces", "registry", "orchestration", "planning")
ROUTER_SPECIFIC_NAMES = (
    "JevRouter",
    "LLMRouter",
    "FallbackRouter",
    "ModelRouter",
    "RoutingRequest",
    "LLMAdapter",
    "RoutingModelAdapter",
    "FakeJevAdapter",
    "FakeLLMAdapter",
    "jevpilot.routing",
    "jevpilot.adapters",
)


def test_generic_orchestration_does_not_name_routing_implementations() -> None:
    hits = []
    for p in CORE_FILES:
        if _subpackage(p) not in ROUTER_AGNOSTIC:
            continue
        code = _code_without_docstrings(p)
        hits += [f"{p.relative_to(ROOT)}: {n}" for n in ROUTER_SPECIFIC_NAMES if n in code]
    assert not hits, "routing implementation leaked into generic orchestration:\n" + "\n".join(hits)


def test_routing_never_depends_on_concrete_adapters() -> None:
    for p in CORE_FILES:
        if _subpackage(p) == "routing":
            for name in _imports(p):
                assert not name.startswith("jevpilot.adapters"), f"{p} imports {name}"


def _code_without_docstrings(path: Path) -> str:
    """Source with docstrings removed, so prose may mention routers but code may not."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0].value.value = ""
    return ast.unparse(tree)
