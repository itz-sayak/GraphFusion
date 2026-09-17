"""OpenAI-compatible chat-completions provider.

Covers NVIDIA NIM, xAI Grok, Ollama and any other endpoint implementing the
``/v1/chat/completions`` function-calling protocol.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from backend.core.errors import LLMProviderError
from backend.llm.base import LLMProvider, LLMResponse, ToolCall
from backend.llm.rate_limit import RateLimiter, RateLimitExhausted, estimate_tokens, parse_duration
from backend.logging_conf import get_logger

log = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0, extra_body: dict[str, Any] | None = None, retries: int = 2, stream: bool = False,
                 fallback_models: list[str] | None = None, model_params: dict[str, dict[str, Any]] | None = None, limiter: RateLimiter | None = None):
        if not base_url or not model:
            raise LLMProviderError(f"{self.name}: base_url and model are required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.retries = retries
        self.stream = stream
        # rate-limit aware failover: limits are per model, so another model usually still has budget
        self.models = [model] + [m for m in (fallback_models or []) if m and m != model]
        self.model_params = model_params or {}
        self.limiter = limiter
        self.last_model = model
        self.long_wait_s = 75.0

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.last_model, "configured_model": self.model, "fallback_models": self.models[1:]}

    def budget_ok(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 256) -> bool:
        """True if a call of this size fits some model's limits right now, without waiting."""
        if self.limiter is None:
            return True
        est = estimate_tokens({"m": messages, "t": tools or []}, expected_output=min(max_tokens, 300))
        return any(self.limiter.headroom(m, est) for m in self.models)

    def _body(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, temperature: float, max_tokens: int) -> dict[str, Any]:
        params = {**self.extra_body, **self.model_params.get(model, {})}
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **params}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        return body

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, temperature: float = 0.1, max_tokens: int = 2048,
             max_wait_s: float | None = None, _second_pass: bool = False) -> LLMResponse:
        """``max_wait_s``: how long the *primary* model may wait for its per-minute window (None = limiter default,
        0 = never wait). Fallback models are tried without waiting. If every model is only minute-limited and the
        soonest slot is within ``long_wait_s``, wait for it once instead of failing."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.stream:
            return self._chat_stream(self._body(self.model, messages, tools, temperature, max_tokens), headers)
        exhausted: list[RateLimitExhausted] = []
        last_exc: Exception | None = None
        for i, model in enumerate(self.models):
            wait_budget = max_wait_s if i == 0 else 0.0
            body = self._body(model, messages, tools, temperature, max_tokens)
            est = estimate_tokens(body, expected_output=min(int(body.get("max_tokens", max_tokens)), 300))
            for attempt in range(self.retries + 1):
                ticket = None
                if self.limiter is not None:
                    try:
                        ticket = self.limiter.acquire(model, est, wait_budget)
                    except RateLimitExhausted as exc:
                        exhausted.append(exc)
                        log.info("Model skipped: rate-limit budget", model=model, reason=exc.message)
                        break
                try:
                    started = time.perf_counter()
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                except httpx.HTTPError as exc:
                    if ticket and self.limiter:
                        self.limiter.release(model, ticket)
                    last_exc = exc
                    if attempt < self.retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                if resp.status_code == 429:
                    if ticket and self.limiter:
                        self.limiter.release(model, ticket)
                        secs = self.limiter.block_from_429(model, resp.text, dict(resp.headers))
                        exhausted.append(RateLimitExhausted(f"{model}: HTTP 429 ({resp.text[:160]})", time.time() + secs, "minute" if secs < 120 else "day"))
                        break  # blocked until the provider's retry time: fail over instead of hammering
                    if attempt < self.retries:
                        time.sleep(self._retry_delay(resp, attempt))
                        continue
                    last_exc = LLMProviderError(f"{self.name} returned HTTP 429: {resp.text[:400]}")
                    break
                if resp.status_code in (500, 502, 503, 504):
                    if ticket and self.limiter:
                        self.limiter.release(model, ticket)
                    last_exc = LLMProviderError(f"{self.name} returned HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < self.retries:
                        time.sleep(self._retry_delay(resp, attempt))
                        continue
                    break
                if resp.status_code >= 400:
                    if ticket and self.limiter:
                        self.limiter.record(model, ticket, est, dict(resp.headers))
                    raise LLMProviderError(f"{self.name} returned HTTP {resp.status_code}: {resp.text[:400]}")
                try:
                    data = resp.json()
                except json.JSONDecodeError as exc:
                    last_exc = exc
                    break
                usage = data.get("usage") or {}
                if ticket and self.limiter:
                    self.limiter.record(model, ticket, usage.get("total_tokens"), dict(resp.headers))
                self.last_model = model
                log.info("LLM call", provider=self.name, model=model, elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                         usage={k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}, estimate=est)
                return self._parse(data)
        if exhausted and last_exc is None:
            soonest = min(exhausted, key=lambda e: e.retry_at)
            delay = soonest.retry_at - time.time()
            if not _second_pass and max_wait_s != 0 and soonest.scope == "minute" and delay <= self.long_wait_s:
                log.info("All models minute-limited; waiting for the next slot", wait_s=round(delay, 1))
                time.sleep(max(0.0, delay) + 0.5)
                return self.chat(messages, tools, temperature, max_tokens, max_wait_s=5.0, _second_pass=True)
            raise RateLimitExhausted("all models are at their rate limits; " + "; ".join(e.message for e in exhausted), soonest.retry_at, soonest.scope)
        raise LLMProviderError(f"{self.name} request failed: {last_exc}")

    @staticmethod
    def _retry_delay(resp: httpx.Response, attempt: int) -> float:
        """Honour Retry-After (rate limits) when the server sends it, else exponential backoff."""
        ra = resp.headers.get("retry-after")
        try:
            return min(60.0, float(ra)) if ra else min(30.0, 2.0 * (2 ** attempt))
        except ValueError:
            return min(30.0, 2.0 * (2 ** attempt))

    def _chat_stream(self, body: dict[str, Any], headers: dict[str, str]) -> LLMResponse:
        """Server-sent-events streaming with tool-call delta accumulation.

        ``timeout`` is a *total* latency budget for the call: queued reasoning
        models can hold a request for minutes, and the agent prefers to fall
        back to the rule-based parser rather than leave the user waiting.
        """
        body = {**body, "stream": True}
        headers = {**headers, "Accept": "text/event-stream"}
        deadline = time.perf_counter() + self.timeout
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=20.0)) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as resp:
                    if resp.status_code >= 400:
                        raise LLMProviderError(f"{self.name} returned HTTP {resp.status_code}: {resp.read()[:400]!r}")
                    for line in resp.iter_lines():
                        if time.perf_counter() > deadline:
                            raise LLMProviderError(f"{self.name} exceeded the {self.timeout:.0f}s latency budget")
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        usage = chunk.get("usage") or usage
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                content.append(delta["content"])
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] = fn.get("name") or slot["name"]
                                slot["arguments"] += fn.get("arguments") or ""
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"{self.name} streaming request failed: {exc}") from exc
        log.info("LLM call", provider=self.name, model=self.model, stream=True, elapsed_ms=round((time.perf_counter() - started) * 1000, 1), usage=usage)
        tool_calls = []
        for _, c in sorted(calls.items()):
            try:
                args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
            except json.JSONDecodeError:
                args = {"__unparseable__": c["arguments"]}
            tool_calls.append(ToolCall(id=c["id"] or f"call_{uuid.uuid4().hex[:8]}", name=c["name"], arguments=args))
        return LLMResponse(content="".join(content) or None, tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"__unparseable__": args}
            calls.append(ToolCall(id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}", name=fn.get("name", ""), arguments=args))
        return LLMResponse(content=msg.get("content"), tool_calls=calls, raw=data, usage=data.get("usage") or {})


class NvidiaNIMProvider(OpenAICompatibleProvider):
    name = "nvidia_nim"


_parse_duration = parse_duration  # backwards-compatible name


class GroqProvider(OpenAICompatibleProvider):
    """Groq LPU inference (OpenAI-compatible endpoint, e.g. openai/gpt-oss-120b)."""

    name = "groq"


class XAIGrokProvider(OpenAICompatibleProvider):
    name = "xai_grok"


class OllamaProvider(OpenAICompatibleProvider):
    """Local models served by Ollama's OpenAI-compatible endpoint."""

    name = "ollama"
