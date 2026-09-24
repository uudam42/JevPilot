"""End-to-end applications built on JevPilot.

An application is the composition root: it chooses the router, the model
adapters (from ``integrations/`` or the offline fakes) and the domain, and
exposes one entry point. The core never imports it, and domains do not depend
on it.
"""
