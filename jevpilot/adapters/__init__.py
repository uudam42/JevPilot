"""Model adapters for the routing layer.

This package ships only provider-neutral, offline adapters: deterministic
fakes for tests, CI, benchmarks, replay and failure injection. Adapters
depend on :mod:`jevpilot.routing` contracts; routing never imports adapters.
Adapters that need a provider SDK live outside the core package (see
``integrations/``), so installing JevPilot never pulls in an SDK.
"""

from jevpilot.adapters import scripting
from jevpilot.adapters.jev import FakeJevAdapter
from jevpilot.adapters.llm import FakeLLMAdapter
from jevpilot.adapters.scripting import ScriptExhaustedError, with_faults

__all__ = [
    "FakeJevAdapter",
    "FakeLLMAdapter",
    "ScriptExhaustedError",
    "scripting",
    "with_faults",
]
