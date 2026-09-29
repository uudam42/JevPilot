"""LangChain as an optional LLM communication / structured-output layer (offline, no network).

The LangChain interpreter must honour the same contract, prompt, schema and
validation as the native interpreter, fail visibly (never fall back), and stay
out of the core and the domain.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from apps.uav_materials import (
    DEMO_REQUEST,
    ComponentUnavailable,
    LiveModeUnavailable,
    RunMode,
    run_uav_material_workflow,
)
from apps.uav_materials import workflow as app_workflow
from domains.uav_materials.interpretation import (
    INTERPRETATION_PROMPT_VERSION,
    InterpretationPayload,
    RequirementInterpreter,
    RuleBasedInterpreter,
    interpretation_system_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
HAVE_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
needs_langchain = pytest.mark.skipif(not HAVE_LANGCHAIN, reason="langchain-core not installed")


def _scripted(payload: Any = None, *, fail: Exception | None = None, tool: bool = True) -> Any:
    """A LangChain chat model answering with a fixed payload (or raising)."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from integrations.langchain_chat import ScriptedToolChatModel

    class Model(ScriptedToolChatModel):
        seen: list[Any] = []

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> Any:
            self.seen.append((messages, kw))
            if fail is not None:
                raise fail
            if tool:
                return super()._generate(messages, stop, run_manager, **kw)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="no tool"))])

    return Model(policy=lambda messages: payload)


def _item(**kw: Any) -> dict[str, Any]:
    base = {
        "aspect": "density",
        "direction": "unspecified",
        "operator": "<=",
        "value": 1800,
        "unit": "kg/m^3",
        "priority": "hard",
        "source_text": "The density must not exceed 1800 kg/m^3.",
    }
    return {**base, **kw}


def _payload(*items: dict[str, Any]) -> dict[str, Any]:
    return {"requirements": list(items), "concerns": [], "application": "", "design_allowed": True}


# -- contract, prompt, schema, validation ----------------------------------------------------------


@needs_langchain
def test_same_contract_prompt_and_schema_as_the_native_interpreter() -> None:
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter
    from apps.uav_materials.llm_interpreter import LLMRequirementInterpreter
    from jevpilot.adapters import FakeLLMAdapter

    model = _scripted(_payload(_item()))
    lc = LangChainRequirementInterpreter(model)
    native = LLMRequirementInterpreter(FakeLLMAdapter(policy=lambda _: {}))
    contract: RequirementInterpreter = lc  # checked by mypy --strict
    assert callable(contract.interpret) and callable(contract.describe)
    assert lc.schema == native.prompt(DEMO_REQUEST).response_schema
    system, user = lc.messages(DEMO_REQUEST)
    assert system.content == native.prompt(DEMO_REQUEST).system == interpretation_system_prompt()
    assert user.content == native.prompt(DEMO_REQUEST).user
    lc.interpret(DEMO_REQUEST)
    (messages, kwargs), *_ = model.seen
    (tool,) = kwargs["tools"]  # structured output requested through tool calling
    assert tool["function"]["name"] == "InterpretationPayload"
    assert lc.describe()["prompt_version"] == INTERPRETATION_PROMPT_VERSION


@needs_langchain
def test_structured_output_goes_through_the_domain_validation() -> None:
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter

    good = _item()
    invented = _item(value=1500, source_text="The density must not exceed 1800 kg/m^3.")
    wrong_unit = _item(aspect="normal_stiffness", direction="laminate_x", operator=">=", value=40,
                       unit="MPa", source_text="stiffness of at least 40 GPa")  # fmt: skip
    outcome = LangChainRequirementInterpreter(
        _scripted(_payload(good, invented, wrong_unit))
    ).interpret(
        "The density must not exceed 1800 kg/m^3. The panel needs a stiffness of at least 40 GPa."
    )
    assert outcome.status == "partial" and len(outcome.requirements) == 1
    assert len(outcome.findings) == 2  # the invented number and the changed unit are dropped
    assert outcome.interpreter["backend"] == "langchain"


@needs_langchain
def test_material_properties_cannot_be_smuggled_in() -> None:
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter

    payload = {**_payload(_item()), "material": {"name": "unobtainium", "density": 1}}
    outcome = LangChainRequirementInterpreter(_scripted(payload)).interpret(
        "The density must not exceed 1800 kg/m^3."
    )
    assert outcome.status == "failed" and "not a valid interpretation" in (outcome.error or "")
    assert "material" not in InterpretationPayload.model_fields


@needs_langchain
def test_incomplete_request_reports_missing_information() -> None:
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter

    text = "I need a lightweight material for a drone."
    wish = {
        "aspect": "density",
        "direction": "unspecified",
        "operator": "minimize",
        "priority": "soft",
        "source_text": text,
    }
    outcome = LangChainRequirementInterpreter(_scripted(_payload(wish))).interpret(text)
    assert outcome.status == "partial" and not outcome.hard
    assert any("No numeric acceptance limit" in m for m in outcome.missing_information)


@needs_langchain
def test_invalid_or_missing_structured_output_fails_visibly() -> None:
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter

    not_a_payload = LangChainRequirementInterpreter(_scripted({"requirements": "many"}))
    assert not_a_payload.interpret("x").status == "failed"
    no_tool_call = LangChainRequirementInterpreter(_scripted(tool=False)).interpret("x")
    assert no_tool_call.status == "failed"
    assert "structured output missing" in (no_tool_call.error or "")


@needs_langchain
def test_provider_errors_are_raised_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.uav_materials.langchain_interpreter import (
        LangChainInterpreterError,
        LangChainRequirementInterpreter,
    )

    secret = "sk-ant-" + "z" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    interpreter = LangChainRequirementInterpreter(_scripted(fail=RuntimeError(f"bad {secret}")))
    with pytest.raises(LangChainInterpreterError) as info:
        interpreter.interpret(DEMO_REQUEST)
    assert secret not in str(info.value)


# -- provider abstraction --------------------------------------------------------------------------


@needs_langchain
def test_provider_spec_parsing_and_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    from integrations.langchain_chat import missing_requirements, parse_spec

    monkeypatch.delenv("JEVPILOT_LANGCHAIN_MODEL", raising=False)
    monkeypatch.delenv("JEVPILOT_ANTHROPIC_MODEL", raising=False)
    assert parse_spec("anthropic:claude-x") == ("anthropic", "claude-x")
    assert parse_spec(None)[0] == "anthropic"
    monkeypatch.setenv("JEVPILOT_LANGCHAIN_MODEL", "anthropic:claude-y")
    assert parse_spec(None) == ("anthropic", "claude-y")
    with pytest.raises(ValueError, match="unknown LangChain provider"):
        parse_spec("nonexistent:model")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert any("ANTHROPIC_API_KEY" in m for m in missing_requirements("anthropic"))


@needs_langchain
def test_real_provider_model_is_built_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_anthropic")
    from apps.uav_materials.langchain_interpreter import LangChainRequirementInterpreter
    from integrations.langchain_chat import chat_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-construction-only")
    model = chat_model("anthropic:claude-test", timeout_s=5)
    described = LangChainRequirementInterpreter(model).describe()
    assert described["kind"] == "llm" and described["chat_model"]["model"] == "claude-test"
    assert "dummy-key" not in json.dumps(described)


# -- workflow: offline demo, live gating, no fallback ----------------------------------------------


@needs_langchain
def test_offline_langchain_demo_reaches_the_same_result_as_the_native_demo() -> None:
    native = run_uav_material_workflow()
    lc = run_uav_material_workflow(llm_backend="langchain")
    assert (lc.mode, lc.decision, lc.report.status) == (RunMode.OFFLINE_DEMO, "design", "complete")
    assert (
        lc.report.interpretation and lc.report.interpretation.interpreter == "langchain_interpreter"
    )
    strip = {"run", "interpretation", "limitations"}
    assert lc.report.model_dump(exclude=strip) == native.report.model_dump(exclude=strip)
    assert any("scripted chat model" in x for x in lc.report.limitations)
    assert not any("interpreted by a language model" in x for x in lc.report.limitations)


def test_native_backend_is_unchanged_and_does_not_import_langchain() -> None:
    code = (
        "import sys\n"
        "from apps.uav_materials import run_uav_material_workflow\n"
        "r = run_uav_material_workflow()\n"
        "assert r.decision == 'design'\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0].startswith('langchain')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_missing_langchain_is_a_clear_error_not_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> Any:
        return None if name.startswith("langchain") else real(name, *args)

    monkeypatch.setattr(app_workflow.importlib.util, "find_spec", find_spec)
    with pytest.raises(ComponentUnavailable, match="langchain"):
        run_uav_material_workflow(llm_backend="langchain")
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy")
    assert any("langchain" in m for m in app_workflow.live_requirements("langchain"))


def test_live_langchain_without_credentials_never_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for mode in (RunMode.LIVE, RunMode.LIVE_ROUTING):
        with pytest.raises(LiveModeUnavailable, match="Nothing was run"):
            run_uav_material_workflow(mode=mode, llm_backend="langchain")


@needs_langchain
def test_offline_langchain_path_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    assert run_uav_material_workflow(llm_backend="langchain").complete


def test_cli_accepts_the_backend_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from apps.cli import main

    if not HAVE_LANGCHAIN:
        pytest.skip("langchain-core not installed")
    assert main(["uav-materials", "--demo", "--llm-backend", "langchain", "--quiet"]) == 0
    assert "[OFFLINE_DEMO] decision: design" in capsys.readouterr().err


# -- boundaries ------------------------------------------------------------------------------------


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "path",
    [
        *sorted((ROOT / "jevpilot").rglob("*.py")),
        *sorted((ROOT / "domains").rglob("*.py")),
        ROOT / "integrations" / "typesafe_jev.py",
        ROOT / "integrations" / "anthropic_llm.py",
        *sorted((ROOT / "experiments").rglob("*.py")),
    ],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_langchain_stays_out_of_core_domain_routing_and_science(path: Path) -> None:
    bad = {n for n in _imports(path) if n.split(".")[0].startswith(("langchain", "langgraph"))}
    assert not bad, f"{path} imports {bad}"


def test_langgraph_is_not_a_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "langgraph" not in text and '"langchain>' not in text


def test_rule_based_interpreter_is_what_the_scripted_model_serves() -> None:
    if not HAVE_LANGCHAIN:
        pytest.skip("langchain-core not installed")
    from apps.uav_materials.langchain_interpreter import offline_langchain_interpreter

    a = offline_langchain_interpreter().interpret(DEMO_REQUEST)
    b = RuleBasedInterpreter().interpret(DEMO_REQUEST)
    assert [r.model_dump() for r in a.requirements] != [] and len(a.requirements) == len(
        b.requirements
    )
    assert [r.label for r in a.requirements] == [r.label for r in b.requirements]
