"""Runtime settings (environment) and algorithm configuration (YAML).

Environment settings control *where* and *with what* the system runs.
YAML configuration controls *how* the algorithms behave: every weight and
threshold is read from there so experiments can vary them without code edits.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    dfg_workspace: Path = PROJECT_ROOT / "workspace"
    dfg_config: Path = CONFIG_DIR / "config.yaml"
    dfg_log_level: str = "INFO"
    dfg_log_json: bool = False
    dfg_cors_origins: str = "http://localhost:3000"
    # optional multi-user access: "alice:token1,bob:token2". Empty = no authentication (single-user local use)
    dfg_api_tokens: str = ""
    dfg_embedding_backend: str = "tfidf"

    dfg_llm_provider: str = "mock"
    nvidia_api_key: str = ""
    dfg_nim_model: str = "openai/gpt-oss-120b"
    dfg_nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    dfg_nim_reasoning_effort: str = "low"
    # total latency budget per LLM call; on expiry the agent falls back to the rule-based parser
    dfg_llm_timeout: float = 90.0
    groq_api_key: str = ""
    dfg_groq_model: str = "qwen/qwen3.8-27b"
    # tried in order when the primary model is at a rate limit (Groq limits are per model)
    dfg_groq_fallback_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b"
    dfg_groq_reasoning_effort: str = "medium"
    xai_api_key: str = ""
    dfg_grok_model: str = "grok-4"
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""
    anthropic_api_key: str = ""
    dfg_anthropic_model: str = "claude-sonnet-5"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"

    census_api_key: str = ""

    @property
    def api_tokens(self) -> dict[str, str]:
        """token → user id"""
        out = {}
        for item in self.dfg_api_tokens.split(","):
            if ":" in item:
                user, token = item.split(":", 1)
                if user.strip() and token.strip():
                    out[token.strip()] = user.strip()
        return out

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.dfg_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache
def _base_config() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.dfg_config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = _load_yaml(path)
    cfg["embeddings"]["backend"] = settings.dfg_embedding_backend or cfg["embeddings"]["backend"]
    return cfg


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a deep copy of the algorithm config with optional nested overrides."""
    cfg = copy.deepcopy(_base_config())
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


@lru_cache
def value_maps() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "value_maps.yaml")


@lru_cache
def abbreviations() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "abbreviations.yaml")


@lru_cache
def glossary() -> dict[str, str]:
    path = CONFIG_DIR / "glossary.yaml"
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in (_load_yaml(path).get("columns") or {}).items()}


@lru_cache
def fx_rates() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "fx_rates.yaml")
