"""Provider-agnostic LLM interface.

The LLM's only job is to translate a user's request into calls of the
registered tools (and to phrase replies). Tools are executed by the backend
after validation; the model never runs code and never touches data directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        """``messages`` use the OpenAI chat format; ``tools`` are JSON-schema function specs."""

    @property
    def is_mock(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "mock": self.is_mock}
