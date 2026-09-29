"""Keep credentials out of messages, logs and result files.

Provider SDKs already avoid echoing keys, but an error message can still carry
text from somewhere unexpected (a proxy, a misconfigured base URL, a user's
own exception). Every error message and log record produced by the provider
integrations passes through :func:`redact` before it is stored or shown.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

# Environment variables whose values are credentials. Their names may be shown; values never.
SECRET_ENV_VARS: tuple[str, ...] = (
    "TYPESAFE_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
)
MASK = "[REDACTED]"
_MIN_SECRET_LENGTH = 8  # shorter values would redact ordinary words
_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(authorization|x-api-key|api[_-]?key)(\s*[:=]\s*)['\"]?[^\s'\",}]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
)


def secret_values(names: Iterable[str] = SECRET_ENV_VARS) -> list[str]:
    """The current values of the credential variables (for matching only; never shown)."""
    values = (os.environ.get(n, "").strip() for n in names)
    return sorted((v for v in values if len(v) >= _MIN_SECRET_LENGTH), key=len, reverse=True)


def redact(text: object, *, extra: Iterable[str] = ()) -> str:
    """``text`` with credential values and credential-shaped tokens replaced by ``[REDACTED]``."""
    out = str(text)
    for value in [*secret_values(), *(e for e in extra if len(e) >= _MIN_SECRET_LENGTH)]:
        out = out.replace(value, MASK)
    out = _PATTERNS[0].sub(lambda m: f"{m.group(1)} {MASK}", out)
    out = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", out)
    return _PATTERNS[2].sub(MASK, out)


def contains_secret(text: str) -> bool:
    """Whether ``text`` contains a configured credential value (used by tests and scans)."""
    return any(v in text for v in secret_values())
