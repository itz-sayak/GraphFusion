"""Provider selection from environment settings."""
from __future__ import annotations

from backend.config import Settings, get_settings
from backend.core.errors import LLMProviderError
from backend.llm.base import LLMProvider
from backend.llm.providers.anthropic import AnthropicProvider
from backend.llm.providers.mock import MockProvider
from backend.llm.providers.openai_compat import GroqProvider, NvidiaNIMProvider, OllamaProvider, OpenAICompatibleProvider, XAIGrokProvider

PROVIDERS = ("mock", "groq", "nvidia_nim", "xai_grok", "openai_compat", "anthropic", "ollama")


def make_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    name = (name or s.dfg_llm_provider or "mock").lower()
    if name == "mock":
        return MockProvider()
    if name == "groq":
        if not s.groq_api_key:
            raise LLMProviderError("groq selected but GROQ_API_KEY is empty")
        from backend.config import load_config
        from backend.llm.rate_limit import shared_limiter

        extra = {"reasoning_effort": s.dfg_groq_reasoning_effort} if s.dfg_groq_reasoning_effort else {}
        llm_cfg = load_config().get("llm") or {}
        return GroqProvider("https://api.groq.com/openai/v1", s.groq_api_key, s.dfg_groq_model, timeout=s.dfg_llm_timeout, extra_body={**extra, "max_tokens": 1024}, retries=2,
                            fallback_models=[m.strip() for m in s.dfg_groq_fallback_models.split(",") if m.strip()],
                            model_params=llm_cfg.get("model_params") or {}, limiter=shared_limiter())
    if name == "nvidia_nim":
        if not s.nvidia_api_key:
            raise LLMProviderError("nvidia_nim selected but NVIDIA_API_KEY is empty")
        # reasoning models on NIM accept an effort hint; low effort keeps tool routing responsive
        extra = {"reasoning_effort": s.dfg_nim_reasoning_effort} if s.dfg_nim_reasoning_effort else {}
        return NvidiaNIMProvider(s.dfg_nim_base_url, s.nvidia_api_key, s.dfg_nim_model, timeout=s.dfg_llm_timeout, extra_body=extra, retries=0, stream=True)
    if name == "xai_grok":
        if not s.xai_api_key:
            raise LLMProviderError("xai_grok selected but XAI_API_KEY is empty")
        return XAIGrokProvider("https://api.x.ai/v1", s.xai_api_key, s.dfg_grok_model, timeout=s.dfg_llm_timeout)
    if name == "openai_compat":
        return OpenAICompatibleProvider(s.openai_compat_base_url, s.openai_compat_api_key, s.openai_compat_model)
    if name == "anthropic":
        return AnthropicProvider(s.anthropic_api_key, s.dfg_anthropic_model)
    if name == "ollama":
        return OllamaProvider(s.ollama_base_url, "", s.ollama_model)
    raise LLMProviderError(f"Unknown LLM provider {name!r}; choose one of {PROVIDERS}")
