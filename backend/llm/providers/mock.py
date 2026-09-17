"""Deterministic mock provider — the rule-based parser behind the LLMProvider interface.

It lets every conversational feature (and the test-suite) run without any API
key, and exercises exactly the same tool-execution path as a real model.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from backend.llm.base import LLMProvider, LLMResponse, ToolCall


class MockProvider(LLMProvider):
    name = "mock"
    model = "rule-based-intent-parser"

    def __init__(self) -> None:
        self.session = None  # bound by the agent for the duration of a turn

    @property
    def is_mock(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        from backend.llm.nl_fallback import parse

        if messages and messages[-1]["role"] == "tool":
            return LLMResponse(content=None)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        calls = parse(self.session, user) if self.session is not None else []
        return LLMResponse(content=None, tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=n, arguments=a) for n, a in calls],
                           raw={"parsed": json.loads(json.dumps(calls, default=str))})
