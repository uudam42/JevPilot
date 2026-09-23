"""Jev (TypeSafe "System One") as a :class:`~jevpilot.routing.contracts.RoutingModelAdapter`.

Built only against the public ``typesafe-sdk`` (verified with 0.7.1). See
``docs/REAL_ROUTING.md`` for the findings this adapter relies on. Install with
``pip install 'jevpilot[jev]'`` and set ``TYPESAFE_API_KEY``. The model comes
from the ``model`` argument, else ``TYPESAFE_DEFAULT_MODEL``, else the SDK
default (``jev-latest``).

Jev does not generate text. It answers typed questions about a *state*:
``Choice`` (at most 255 labels; returns the label, a confidence and the full
probability map), ``Noul`` (probability of "yes") and ``Score`` (expected level
on a rubric). The canonical :class:`RoutingRequest` is mapped onto those
questions as follows (request format :data:`REQUEST_FORMAT_VERSION`):

1. **Next action.** One ``Choice`` whose labels are the offered capability ids
   plus ``__finish__`` and ``__ask_human__``. Each label's criterion is that
   capability's full description (description, schemas, preconditions,
   effects, tags, cost). The rest of the request is the state. The
   information is the same as the LLM prompt; only its layout differs.
2. **Inputs** (only when the chosen capability has required inputs): a second
   call with one question per required field. ``enum`` fields become a
   ``Choice`` over the enum values, ``boolean`` a ``Noul``, a bounded
   ``integer`` (≤ 256 levels) a ``Score``. Any other field becomes a ``Choice``
   over the scalar values present in the state (goal, context, state
   extensions), because **Jev cannot produce a value that is not already
   written somewhere in the state**. This is a structural difference from
   text-generating routers and is reported as such.

The result is emitted as the standard decision payload and goes through the
same strict parsing and validation as every other model router. Jev's
reported confidence and the full probability distribution are recorded in
the attempt metadata. The model reports the served ``model`` name and token
``usage``. Jev exposes no sampling controls.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from typing import Any

from jevpilot.core import ModelUsage
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError
from jevpilot.routing.contracts import RoutingModelResponse
from jevpilot.routing.request import CapabilityDescription, RoutingRequest

ADAPTER_VERSION = "typesafe-jev-adapter/1"
REQUEST_FORMAT_VERSION = "jevpilot-jev-questions/1"
PROVIDER = "typesafe"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
SDK_DEFAULT_MODEL = "jev-latest"
FINISH, ASK_HUMAN, NONE = "__finish__", "__ask_human__", "__none__"
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 256

NEXT_ACTION_INSTRUCTIONS = (
    "You are the routing policy of a workflow orchestration system. Given the workflow state "
    "(goal, context, previous actions and their results or errors, evaluations, plan), choose "
    "the single next action. Choose a capability only if its preconditions hold in the current "
    "state and invoking it moves the workflow toward the goal; avoid unnecessary actions and do "
    "not repeat a step that already succeeded. Choose __finish__ if the goal is already met. "
    "Choose __ask_human__ if progress needs information that only a human can provide and no "
    "available capability can obtain it."
)
FINISH_CRITERION = "The goal is already achieved. Stop without invoking any capability."
ASK_HUMAN_CRITERION = (
    "Progress needs information that only a human can provide, and no available capability "
    "can obtain it. Stop and ask a human."
)


class TypeSafeJevAdapter:
    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        max_retries: int = 0,
        price_per_mtok_input: float | None = None,
        pricing_source: str | None = None,
    ) -> None:
        self._sdk: Any = _optional_sdk()
        self.model = model or os.environ.get(MODEL_ENV, "").strip() or SDK_DEFAULT_MODEL
        self.max_retries = max_retries
        self.price_per_mtok_input = price_per_mtok_input
        self.pricing_source = pricing_source
        if client is None:
            if self._sdk is None:
                raise ImportError(
                    "TypeSafeJevAdapter needs the TypeSafe SDK: pip install 'jevpilot[jev]'"
                )
            # Raises the SDK's TypeSafeError when TYPESAFE_API_KEY is missing.
            client = self._sdk.TypeSafeClient(
                model=self.model, retry=self._sdk.RetryPolicy(max_retries=max_retries)
            )
        self._client = client

    # -- RoutingModelAdapter -------------------------------------------------------

    def infer(
        self, request: RoutingRequest, *, timeout_s: float | None = None
    ) -> RoutingModelResponse:
        caps = {c.id: c for c in request.available_capabilities}
        if len(caps) + 2 > MAX_CHOICES:
            raise RouterAdapterError(
                f"{len(caps)} capabilities exceed Jev's {MAX_CHOICES}-label choice limit",
                retryable=False,
                details=self._identity([]),
            )
        state = request.model_dump(mode="json", exclude={"available_capabilities"})
        criteria: dict[str, Any] = {
            cid: cap.model_dump(mode="json", exclude={"id"}) for cid, cap in caps.items()
        }
        criteria[FINISH] = FINISH_CRITERION
        criteria[ASK_HUMAN] = ASK_HUMAN_CRITERION
        first = self._ask(
            state,
            {
                "next_action": {
                    "type": "choice",
                    "instructions": NEXT_ACTION_INSTRUCTIONS,
                    "criteria": criteria,
                }
            },
            timeout_s,
        )
        answer = first.answers["next_action"]
        label = answer.choice
        responses = [first]
        metadata: dict[str, Any] = {
            "request_format": REQUEST_FORMAT_VERSION,
            "next_action_probabilities": dict(answer.probabilities),
            "next_action_confidence": answer.confidence,
        }
        payload: dict[str, Any] = {
            "reason": "",  # Jev returns no text
            "confidence": answer.confidence,
        }
        if label == FINISH:
            payload.update(action="finish", capability_id=None, inputs={})
        elif label == ASK_HUMAN:
            payload.update(action="ask_human", capability_id=None, inputs={})
        else:
            # An unknown label passes through untouched: validation then rejects it explicitly.
            cap = caps.get(label)
            inputs: dict[str, Any] = {}
            if cap is not None:
                plan = list(_input_questions(cap, request))
                if plan:
                    second = self._ask(
                        {**state, "selected_capability": label},
                        {q.name: q.question for q in plan},
                        timeout_s,
                    )
                    responses.append(second)
                    inputs, answers = _decode_inputs(plan, second)
                    metadata["input_answers"] = answers
            payload.update(action="invoke", capability_id=label, inputs=inputs)
        metadata.update(self._identity(responses))
        return RoutingModelResponse(
            output=payload,
            model=responses[0].model,
            usage=self._usage(responses),
            metadata=metadata,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "kind": "jev",
            "adapter_version": ADAPTER_VERSION,
            "request_format": REQUEST_FORMAT_VERSION,
            "sdk_version": getattr(self._sdk, "__version__", None),
            "model": self.model,
            "mode": "strict",  # the SDK has no server-side model substitution
            "sdk_retries": self.max_retries,
            "sampling": "not exposed by the API",
            "pricing": (
                {
                    "usd_per_mtok_input": self.price_per_mtok_input,
                    "source": self.pricing_source,
                    "output_tokens": "free of charge per API docs",
                }
                if self.price_per_mtok_input is not None
                else None
            ),
        }

    def verify(self) -> dict[str, Any]:
        """Confirm the configured model is listed for these credentials."""
        try:
            listed = self._client.models.list()
        except Exception as exc:
            raise self._translate(exc) from exc
        names = [m.name for m in listed.models]
        if self.model not in names:
            raise RouterAdapterError(
                f"Jev model {self.model!r} is not available; listed: {names}", retryable=False
            )
        meta = next(m for m in listed.models if m.name == self.model)
        return {
            "provider": PROVIDER,
            "requested_model": self.model,
            "listed_models": names,
            "description": meta.description,
            "release_date": meta.release_date,
        }

    # -- helpers ------------------------------------------------------------------

    def _ask(self, state: Any, questions: Mapping[str, Any], timeout_s: float | None) -> Any:
        try:
            return self._client.system_one(
                state=state, questions=dict(questions), model=self.model, timeout=timeout_s
            )
        except Exception as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> Exception:
        details = self._identity([])
        if isinstance(exc, TimeoutError):  # TypeSafeAPITimeoutError subclasses TimeoutError
            return RouterTimeoutError(f"Jev request timed out: {exc}", details=details)
        status = getattr(exc, "status", None)  # TypeSafeAPIError.status
        retryable = isinstance(exc, ConnectionError) or (
            status is not None and (status == 429 or status >= 500)
        )
        details["status_code"] = status
        details["request_id"] = getattr(exc, "request_id", None)
        return RouterAdapterError(
            f"Jev request failed: {type(exc).__name__}: {exc}",
            retryable=retryable,
            details=details,
        )

    def _identity(self, responses: list[Any]) -> dict[str, Any]:
        served = sorted({r.model for r in responses})
        return {
            "provider": PROVIDER,
            "requested_model": self.model,
            "actual_model": served[0] if len(served) == 1 else (served or None),
            "adapter_version": ADAPTER_VERSION,
            "model_calls": len(responses),
        }

    def _usage(self, responses: list[Any]) -> ModelUsage:
        tokens_in = [r.usage.input_tokens for r in responses if r.usage.input_tokens is not None]
        tokens_out = [r.usage.output_tokens for r in responses if r.usage.output_tokens is not None]
        total_in = sum(tokens_in) if tokens_in else None
        cost = None
        if self.price_per_mtok_input is not None and total_in is not None:
            cost = total_in * self.price_per_mtok_input / 1_000_000
        return ModelUsage(
            model_calls=len(responses),
            input_tokens=total_in,
            output_tokens=sum(tokens_out) if tokens_out else None,
            estimated_cost_usd=cost,
        )


# -- input questions ---------------------------------------------------------------


class _FieldQuestion:
    def __init__(
        self,
        name: str,
        field: str,
        kind: str,
        question: dict[str, Any],
        values: dict[str, Any] | None = None,
        low: int = 0,
    ) -> None:
        self.name, self.field, self.kind = name, field, kind
        self.question, self.values, self.low = question, values or {}, low


def _input_questions(
    cap: CapabilityDescription, request: RoutingRequest
) -> Iterator[_FieldQuestion]:
    schema = cap.input_schema or {}
    props: dict[str, Any] = schema.get("properties", {})
    for i, field in enumerate(schema.get("required", [])):
        prop = _resolve(props.get(field, {}), schema)
        name = f"input_{i}"
        base = (
            f"Value for input field '{field}' of capability '{cap.id}' "
            f"(schema: {prop}). Use the workflow state."
        )
        if "enum" in prop and 0 < len(prop["enum"]) <= MAX_CHOICES:
            values = {str(v): v for v in prop["enum"]}
            yield _FieldQuestion(
                name,
                field,
                "choice",
                {"type": "choice", "instructions": base, "criteria": {k: None for k in values}},
                values,
            )
        elif prop.get("type") == "boolean":
            yield _FieldQuestion(name, field, "noul", {"type": "noul", "instructions": base})
        elif (
            prop.get("type") == "integer"
            and "minimum" in prop
            and "maximum" in prop
            and 0 < prop["maximum"] - prop["minimum"] + 1 <= MAX_SCORE_LEVELS
        ):
            lo, hi = int(prop["minimum"]), int(prop["maximum"])
            yield _FieldQuestion(
                name,
                field,
                "score",
                {
                    "type": "score",
                    "instructions": base,
                    "criteria": [str(v) for v in range(lo, hi + 1)],
                },
                low=lo,
            )
        else:
            candidates = _state_values(request, prop.get("type"))
            values = {f"v{j}": c["value"] for j, c in enumerate(candidates)}
            criteria: dict[str, Any] = {f"v{j}": c for j, c in enumerate(candidates)}
            criteria[NONE] = "None of these values is appropriate for this field."
            yield _FieldQuestion(
                name,
                field,
                "grounded_choice",
                {
                    "type": "choice",
                    "instructions": base + " Pick the state value to use.",
                    "criteria": criteria,
                },
                values,
            )


def _decode_inputs(plan: list[_FieldQuestion], response: Any) -> tuple[dict[str, Any], Any]:
    inputs: dict[str, Any] = {}
    record: dict[str, Any] = {}
    for q in plan:
        answer = response.answers[q.name]
        if q.kind in ("choice", "grounded_choice"):
            record[q.field] = {"label": answer.choice, "confidence": answer.confidence}
            if answer.choice in q.values:
                inputs[q.field] = q.values[answer.choice]
        elif q.kind == "noul":
            record[q.field] = {"p_yes": answer.noul}
            inputs[q.field] = answer.noul >= 0.5
        else:
            record[q.field] = {"score": answer.score, "confidence": answer.confidence}
            inputs[q.field] = q.low + round(answer.score)
    return inputs, record


def _state_values(request: RoutingRequest, json_type: Any) -> list[dict[str, Any]]:
    """Distinct scalar values in the goal, context and state extensions (≤ 250)."""
    wanted: tuple[type, ...]
    if json_type == "string":
        wanted = (str,)
    elif json_type in ("number", "integer"):
        wanted = (int, float)
    else:
        wanted = (str, int, float)
    sources = {
        "goal.parameters": request.goal.get("parameters", {}),
        "goal.success_criteria": request.goal.get("success_criteria", {}),
        "state.context": request.state_summary.context,
        "state.extensions": request.state_summary.extensions,
    }
    seen: list[Any] = []
    out: list[dict[str, Any]] = []
    for root, value in sources.items():
        for path, leaf in _leaves(value, root):
            if isinstance(leaf, bool) or not isinstance(leaf, wanted) or leaf in seen:
                continue
            if json_type == "integer" and isinstance(leaf, float) and not leaf.is_integer():
                continue
            seen.append(leaf)
            out.append({"value": leaf, "found_at": path})
            if len(out) >= MAX_CHOICES - 5:
                return out
    return out


def _leaves(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def _resolve(prop: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Inline a local ``$ref`` (Pydantic emits enums as ``$defs`` references)."""
    ref = prop.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return dict(schema.get("$defs", {}).get(ref.rsplit("/", 1)[-1], {}))
    return prop


def _optional_sdk() -> Any:
    try:
        return importlib.import_module("typesafe_sdk")
    except ImportError:
        return None
