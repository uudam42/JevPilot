"""Jev (TypeSafe "System One") as a :class:`~jevpilot.routing.contracts.RoutingModelAdapter`.

Built only against the public ``typesafe-sdk`` (verified with 0.7.1) and the
HTTP API it wraps (``GET /v1/models``, ``POST /v1/systemone``, bearer
authentication; the SDK and the official API reference agree). See
``docs/REAL_ROUTING.md``. Install with ``pip install 'jevpilot[jev]'`` and set
``TYPESAFE_API_KEY``; the key is read by the SDK from the environment and never
stored, logged or echoed by this adapter.

**Model.** The ``model`` argument, else ``TYPESAFE_DEFAULT_MODEL``, else
*discovered*: the adapter lists the models available to the credentials
(``GET /v1/models``) and pins the most recently released one. Nothing is
hard-coded, and the model actually used is recorded with every decision.

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

**Reliability.** SDK-level retries are off; the adapter retries itself so that
every transport event is recorded (``metadata["transport_events"]``). Rate
limits (429, honouring ``Retry-After``), timeouts, connection errors and
server errors (408, 5xx including 529) are retried at most ``max_retries``
times with exponential backoff, and never beyond the router's deadline.
Authentication (401), permission (403), not-found (404) and invalid-request
(400/422) errors are not retried. A response without a well-formed answer is
rejected as ``malformed_response``. Every error message is redacted.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from integrations.redaction import redact
from jevpilot.core import ModelUsage
from jevpilot.exceptions import RouterAdapterError, RouterTimeoutError, RoutingError
from jevpilot.routing.contracts import RoutingModelResponse
from jevpilot.routing.request import CapabilityDescription, RoutingRequest

ADAPTER_VERSION = "typesafe-jev-adapter/2"
REQUEST_FORMAT_VERSION = "jevpilot-jev-questions/1"
PROVIDER = "typesafe"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
FINISH, ASK_HUMAN, NONE = "__finish__", "__ask_human__", "__none__"
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 256
RETRYABLE_KINDS = frozenset({"rate_limited", "server_error", "timeout", "connection"})
_CONFIDENCE_TOLERANCE = 1e-9  # float round-off above 1.0 is clamped; anything larger is rejected

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


class JevUnavailableError(RuntimeError):
    """The Jev adapter cannot be built here (SDK or credentials missing)."""


class TypeSafeJevAdapter:
    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        backoff_max_s: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        price_per_mtok_input: float | None = None,
        pricing_source: str | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._sdk: Any = _optional_sdk()
        env_model = os.environ.get(MODEL_ENV, "").strip()
        self.model: str | None = model or env_model or None
        self.model_source = "argument" if model else "environment" if env_model else None
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        self.price_per_mtok_input = price_per_mtok_input
        self.pricing_source = pricing_source
        self._listed: list[dict[str, str]] | None = None
        if client is None:
            if self._sdk is None:
                raise JevUnavailableError(
                    "TypeSafeJevAdapter needs the TypeSafe SDK: pip install 'jevpilot[jev]'"
                )
            if not os.environ.get("TYPESAFE_API_KEY", "").strip():
                raise JevUnavailableError("TYPESAFE_API_KEY is not set")
            try:  # retries are done (and recorded) by this adapter, not inside the SDK
                client = self._sdk.TypeSafeClient(retry=self._sdk.RetryPolicy(max_retries=0))
            except Exception as exc:  # e.g. a key with invalid characters
                raise JevUnavailableError(
                    f"TypeSafe client could not be created: {type(exc).__name__}: {redact(exc)}"
                ) from None
        self._client = client

    # -- RoutingModelAdapter -------------------------------------------------------

    def infer(
        self, request: RoutingRequest, *, timeout_s: float | None = None
    ) -> RoutingModelResponse:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        events: list[dict[str, Any]] = []
        try:
            model = self.resolve_model(deadline=deadline, events=events)
            caps = {c.id: c for c in request.available_capabilities}
            if len(caps) + 2 > MAX_CHOICES:
                raise RouterAdapterError(
                    f"{len(caps)} capabilities exceed Jev's {MAX_CHOICES}-label choice limit",
                    retryable=False,
                    details={**self._identity([]), "failure": "unsupported_request"},
                )
            state = request.model_dump(mode="json", exclude={"available_capabilities"})
            criteria: dict[str, Any] = {
                cid: cap.model_dump(mode="json", exclude={"id"}) for cid, cap in caps.items()
            }
            criteria[FINISH] = FINISH_CRITERION
            criteria[ASK_HUMAN] = ASK_HUMAN_CRITERION
            first = self._ask(
                model,
                state,
                {
                    "next_action": {
                        "type": "choice",
                        "instructions": NEXT_ACTION_INSTRUCTIONS,
                        "criteria": criteria,
                    }
                },
                deadline,
                events,
            )
            label, confidence, probabilities = self._choice(first, "next_action")
            responses = [first]
            metadata: dict[str, Any] = {
                "request_format": REQUEST_FORMAT_VERSION,
                "next_action_probabilities": probabilities,
                "next_action_confidence": confidence,
            }
            payload: dict[str, Any] = {
                "reason": "",  # Jev returns no text
                "confidence": confidence,
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
                            model,
                            {**state, "selected_capability": label},
                            {q.name: q.question for q in plan},
                            deadline,
                            events,
                        )
                        responses.append(second)
                        try:
                            inputs, answers = _decode_inputs(plan, second)
                        except (KeyError, AttributeError, TypeError, ValueError) as exc:
                            raise self._malformed(f"input answers: {type(exc).__name__}") from exc
                        metadata["input_answers"] = answers
                payload.update(action="invoke", capability_id=label, inputs=inputs)
        except RoutingError as exc:
            exc.details.setdefault("transport_events", list(events))
            raise
        metadata.update(self._identity(responses))
        metadata["transport_events"] = events
        metadata["retries"] = sum(1 for e in events if e.get("retried"))
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
            "model_source": self.model_source,
            "mode": "strict",  # the SDK has no server-side model substitution
            "retries": {
                "max_retries": self.max_retries,
                "backoff_s": self.backoff_s,
                "backoff_max_s": self.backoff_max_s,
                "retried_kinds": sorted(RETRYABLE_KINDS),
                "bounded_by": "the router deadline (timeout_s)",
                "sdk_retries": 0,
            },
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

    # -- models ---------------------------------------------------------------------

    def list_models(
        self, *, deadline: float | None = None, events: list[dict[str, Any]] | None = None
    ) -> list[dict[str, str]]:
        """The models available to these credentials (``GET /v1/models``), cached."""
        if self._listed is None:
            events = [] if events is None else events
            listed = self._call(
                "models.list",
                lambda remaining: self._client.models.list(timeout=remaining),
                deadline,
                events,
            )
            try:
                self._listed = [
                    {
                        "name": str(m.name),
                        "description": str(m.description),
                        "release_date": str(m.release_date),
                    }
                    for m in listed.models
                ]
            except (AttributeError, TypeError) as exc:
                raise self._malformed(f"model list: {type(exc).__name__}") from exc
        return list(self._listed)

    def resolve_model(
        self, *, deadline: float | None = None, events: list[dict[str, Any]] | None = None
    ) -> str:
        """The configured model, or the most recently released listed model (discovered once)."""
        if self.model is None:
            listed = self.list_models(deadline=deadline, events=events)
            if not listed:
                raise RouterAdapterError(
                    "no Jev models are listed for these credentials",
                    retryable=False,
                    details={**self._identity([]), "failure": "no_models"},
                )
            newest = max(listed, key=lambda m: (m["release_date"], m["name"]))
            self.model, self.model_source = newest["name"], "discovered"
        return self.model

    def verify(self) -> dict[str, Any]:
        """Preflight: authenticate, list models, resolve the model and confirm it is listed."""
        listed = self.list_models()
        model = self.resolve_model()
        names = [m["name"] for m in listed]
        if model not in names:
            raise RouterAdapterError(
                f"Jev model {model!r} is not listed for these credentials; listed: {names}",
                retryable=False,
                details={**self._identity([]), "failure": "model_not_listed"},
            )
        meta = next(m for m in listed if m["name"] == model)
        return {
            "provider": PROVIDER,
            "requested_model": model,
            "model_source": self.model_source,
            "listed_models": names,
            "description": meta["description"],
            "release_date": meta["release_date"],
            "sdk_version": getattr(self._sdk, "__version__", None),
            "authentication": "ok",
        }

    # -- calls ----------------------------------------------------------------------

    def _ask(
        self,
        model: str,
        state: Any,
        questions: Mapping[str, Any],
        deadline: float | None,
        events: list[dict[str, Any]],
    ) -> Any:
        response = self._call(
            "systemone",
            lambda remaining: self._client.system_one(
                state=state, questions=dict(questions), model=model, timeout=remaining
            ),
            deadline,
            events,
        )
        if not isinstance(getattr(response, "answers", None), Mapping):
            raise self._malformed("response has no answers")
        return response

    def _call(
        self,
        endpoint: str,
        send: Callable[[float | None], Any],
        deadline: float | None,
        events: list[dict[str, Any]],
    ) -> Any:
        """One API operation with bounded, recorded retries that never pass the deadline."""
        failures = 0
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise RouterTimeoutError(
                    f"Jev {endpoint}: deadline reached before the request could be sent",
                    details={**self._identity([]), "failure": "timeout"},
                )
            try:
                return send(remaining)
            except Exception as exc:
                event = _classify(exc, endpoint)
                events.append(event)
                failures += 1
                delay = self._delay(failures, event.get("retry_after_s"))
                left = None if deadline is None else deadline - time.monotonic()
                can_retry = (
                    event["kind"] in RETRYABLE_KINDS
                    and failures <= self.max_retries
                    and (left is None or delay < left)
                )
                if not can_retry:
                    raise self._error(exc, event, endpoint) from exc
                event["retried"] = True
                event["backoff_s"] = round(delay, 3)
                self._sleep(delay)

    def _delay(self, failures: int, retry_after_s: float | None) -> float:
        if retry_after_s is not None:
            return max(0.0, retry_after_s)
        return min(self.backoff_max_s, self.backoff_s * 2.0 ** (failures - 1))

    def _choice(self, response: Any, key: str) -> tuple[str, float | None, dict[str, float]]:
        """Label, confidence and probability map of one Choice answer, or a malformed error."""
        answer = response.answers.get(key)
        label = getattr(answer, "choice", None)
        if not isinstance(label, str) or not label:
            raise self._malformed(f"answer {key!r} has no choice label")
        confidence = getattr(answer, "confidence", None)
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, int | float):
                raise self._malformed(f"answer {key!r} has a non-numeric confidence")
            confidence = float(confidence)
            if math.isfinite(confidence) and 1.0 < confidence <= 1.0 + _CONFIDENCE_TOLERANCE:
                confidence = 1.0
            # anything else outside [0, 1] (or NaN) is rejected by the decision parser
        raw = getattr(answer, "probabilities", None) or {}
        try:
            probabilities = {str(k): float(v) for k, v in dict(raw).items()}
        except (TypeError, ValueError) as exc:
            raise self._malformed(f"answer {key!r} has invalid probabilities") from exc
        return label, confidence, probabilities

    def _malformed(self, what: str) -> RouterAdapterError:
        return RouterAdapterError(
            f"malformed Jev response: {what}",
            retryable=False,
            details={**self._identity([]), "failure": "malformed_response"},
        )

    def _error(self, exc: Exception, event: dict[str, Any], endpoint: str) -> RoutingError:
        details = {
            **self._identity([]),
            "failure": event["kind"],
            "status_code": event.get("status"),
            "request_id": event.get("request_id"),
        }
        message = f"Jev {endpoint} failed ({event['kind']}"
        if event.get("status") is not None:
            message += f", HTTP {event['status']}"
        message += f"): {type(exc).__name__}: {redact(exc)[:300]}"
        hint = _HINTS.get(event["kind"])
        if hint:
            message += f". {hint}"
        if event["kind"] == "timeout":
            return RouterTimeoutError(message, details=details)
        return RouterAdapterError(
            message, retryable=event["kind"] in RETRYABLE_KINDS, details=details
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


_HINTS = {
    "authentication": "Check that TYPESAFE_API_KEY holds a valid key (its value is never shown)",
    "permission": "The key is valid but not allowed to use this model or endpoint",
    "not_found": "The model or endpoint does not exist for these credentials",
    "invalid_request": "The API rejected the request (question schema, limits or model name)",
    "rate_limited": "Rate limit reached; retries were bounded by max_retries and the deadline",
}


def _classify(exc: Exception, endpoint: str) -> dict[str, Any]:
    """A redaction-safe record of one failed API attempt (never headers or bodies)."""
    status = getattr(exc, "status", None)
    status = status if isinstance(status, int) else None
    retry_after_ms = getattr(exc, "retry_after_ms", None)
    request_id = getattr(exc, "request_id", None)
    if isinstance(exc, TimeoutError):  # TypeSafeAPITimeoutError subclasses TimeoutError
        kind = "timeout"
    elif isinstance(exc, ConnectionError):  # TypeSafeAPIConnectionError subclasses it
        kind = "connection"
    elif status in (401,):
        kind = "authentication"
    elif status == 403:
        kind = "permission"
    elif status == 404:
        kind = "not_found"
    elif status in (400, 422):
        kind = "invalid_request"
    elif status == 429:
        kind = "rate_limited"
    elif status is not None and (status == 408 or status >= 500):
        kind = "server_error"
    elif status is not None and 200 <= status < 300:
        kind = "malformed_response"  # e.g. the SDK's response validation error
    else:
        kind = "unexpected"
    return {
        "endpoint": endpoint,
        "kind": kind,
        "status": status,
        "error_type": type(exc).__name__,
        "retry_after_s": retry_after_ms / 1000 if isinstance(retry_after_ms, int | float) else None,
        "request_id": request_id if isinstance(request_id, str) else None,
    }


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
