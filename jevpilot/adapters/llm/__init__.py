"""LLM adapters. The contract is :class:`jevpilot.routing.LLMAdapter`."""

from jevpilot.adapters.llm.fake import FakeLLMAdapter, request_from_prompt

__all__ = ["FakeLLMAdapter", "request_from_prompt"]
