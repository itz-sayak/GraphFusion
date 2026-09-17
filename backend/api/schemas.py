"""Request / response models for the REST API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    datasets: list[dict[str, Any]]
    discovered: bool
    preferences: dict[str, Any]
    plan_id: str | None
    merges: list[dict[str, Any]]
    chat_turns: int


class UploadResponse(BaseModel):
    session_id: str
    datasets: list[dict[str, Any]]
    errors: list[dict[str, Any]] = Field(default_factory=list)


class LoadPathRequest(BaseModel):
    paths: list[str] = Field(..., description="Paths relative to the project data/ directory")


class LoadUrlRequest(BaseModel):
    url: str
    name: str
    params: dict[str, Any] | None = None
    page_size: int | None = Field(default=None, ge=1, le=50000)


class DiscoverRequest(BaseModel):
    force: bool = False


class PreferencesRequest(BaseModel):
    mode: str | None = None
    conflict_strategy: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=100)
    clear_threshold: bool = False
    primary_key: str | None = None
    merge_uncertain: bool | None = None
    source_priority: list[str] | None = None
    scorer: str | None = Field(default=None, pattern="^(weighted|learned|blend)$")


class PlanRequest(BaseModel):
    mode: str | None = None
    conflict_strategy: str | None = None
    datasets: list[str] | None = None


class ExecuteRequest(BaseModel):
    plan_id: str | None = None
    write_csv: bool = True


class MappingDecisionRequest(BaseModel):
    left: str
    right: str
    decision: str = Field(..., pattern="^(approved|rejected)$")


class RecordRef(BaseModel):
    dataset: str
    row: int = Field(..., ge=0)


class EntityLabelRequest(BaseModel):
    left: RecordRef
    right: RecordRef
    decision: str = Field(..., pattern="^(match|non_match)$")
    sampling: str = Field(default="uncertainty", pattern="^(uncertainty|random)$")


class EntityPreviewRequest(BaseModel):
    left: str
    right: str | None = None
    limit: int = Field(default=20, ge=1, le=200)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    provider: str | None = Field(default=None, description="Override the configured provider for this turn (e.g. 'mock')")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    suggestions: list[str]
    provider: dict[str, Any]
    elapsed_ms: float
