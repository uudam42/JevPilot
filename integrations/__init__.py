"""Provider integrations that need third-party SDKs.

Like ``domains/``, this package depends on JevPilot, and JevPilot never
imports it. Each module here implements an adapter contract from
:mod:`jevpilot.routing.contracts` and imports its SDK lazily, so the core
installs and runs without any provider package.
"""
