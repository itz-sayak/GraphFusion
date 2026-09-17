"""Anthropic Messages API provider (tool use), translated to/from the OpenAI message format."""
from __future__ import annotations

import json
from typing import Any

import httpx

from backend.core.errors import LLMProviderError
from backend.llm.base import LLMProvider, LLMResponse, ToolCall


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.anthropic.com/v1", timeout: float = 120.0):
        if not api_key:
            raise LLMProviderError("anthropic: ANTHROPIC_API_KEY is not set")
        self.api_key, self.model, self.base_url, self.timeout = api_key, model, base_url.rstrip("/"), timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, temperature: float = 0.1, max_tokens: int = 2048) -> LLMResponse:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        converted: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                content: list[dict[str, Any]] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": json.loads(tc["function"]["arguments"] or "{}")})
                converted.append({"role": "assistant", "content": content})
            else:
                converted.append({"role": m["role"], "content": m["content"] or ""})
        body: dict[str, Any] = {"model": self.model, "system": system, "messages": converted, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/messages", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"anthropic request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMProviderError(f"anthropic returned HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text") or None
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {}) for b in data.get("content", []) if b.get("type") == "tool_use"]
        return LLMResponse(content=text, tool_calls=calls, raw=data, usage=data.get("usage") or {})
