"""Core domain models shared by every pipeline stage.

These are Pydantic models so they serialise directly into API responses,
exports (``merge_report.json``, ``schema_mapping.json``) and session snapshots.
Heavy data (tables, sketches) is never stored on these models; they carry
references (paths) and small summaries only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- ingestion


class SourceType(str, Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    XLSX = "xlsx"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    REST = "rest"


class DataType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    OTHER = "other"


class SemanticType(str, Enum):
    ID = "ID"
    NAME = "NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    DATE = "DATE"
    DATETIME = "DATETIME"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    ZIPCODE = "ZIPCODE"
    CURRENCY = "CURRENCY"
    CURRENCY_CODE = "CURRENCY_CODE"
    LATITUDE = "LATITUDE"
    LONGITUDE = "LONGITUDE"
    CATEGORY = "CATEGORY"
    FREE_TEXT = "FREE_TEXT"
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    UNKNOWN = "UNKNOWN"


class ColumnProfile(BaseModel):
    name: str
    physical_type: str
    data_type: DataType
    semantic_type: SemanticType = SemanticType.UNKNOWN
    semantic_confidence: float = 0.0
    semantic_evidence: dict[str, float] = Field(default_factory=dict)
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: int = 0
    uniqueness: float = 0.0  # unique_count / non-null count
    cardinality: str = "unknown"  # unique | high | medium | low | constant
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    quantiles: list[float] | None = None
    avg_length: float | None = None
    sample_values: list[Any] = Field(default_factory=list)
    top_values: list[tuple[Any, int]] = Field(default_factory=list)
    pattern_histogram: dict[str, float] = Field(default_factory=dict)
    date_format: str | None = None
    date_dayfirst: bool | None = None


class DatasetProfile(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    profiled_at: datetime = Field(default_factory=utcnow)
    elapsed_ms: float = 0.0

    def column(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


class DatasetArtifact(BaseModel):
    """Common internal representation of any ingested source."""

    dataset_id: str
    source_name: str
    source_type: SourceType
    source_uri: str
    storage_path: str  # immutable normalised Parquet copy inside the workspace
    schema_: dict[str, str] = Field(default_factory=dict, alias="schema")
    row_count: int = 0
    column_count: int = 0
    checksum: str = ""
    ingested_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    profile: DatasetProfile | None = None

    model_config = {"populate_by_name": True}

    @property
    def schema(self) -> dict[str, str]:  # type: ignore[override]
        return self.schema_

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "schema": self.schema_,
            "ingested_at": self.ingested_at.isoformat(),
            "profiled": self.profile is not None,
        }


# --------------------------------------------------------------------------- matching


class ConfidenceBand(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MatchEvidence(BaseModel):
    """Per-signal breakdown behind a column correspondence score."""

    name_similarity: float = 0.0
    datatype_similarity: float = 0.0
    semantic_similarity: float = 0.0
    value_overlap: float = 0.0
    jaccard: float = 0.0
    containment_left: float = 0.0  # |A ∩ B| / |A|
    containment_right: float = 0.0  # |A ∩ B| / |B|
    distribution_similarity: float = 0.0
    pattern_similarity: float = 0.0
    cardinality_similarity: float = 0.0
    normalizer: str = "identity"
    base_score: float = 0.0
    transitive_support: float = 0.0
    structural_support: float = 0.0
    exclusivity: float = 1.0
    graph_adjustment: float = 0.0
    learned_probability: float | None = None
    row_agreement: float | None = None  # share of key-joined rows on which the two columns agree (graph/verification.py)
    rows_compared: int | None = None
    notes: list[str] = Field(default_factory=list)


class ColumnRef(BaseModel):
    dataset_id: str
    column: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}::{self.column}"

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.key)


class ColumnMatch(BaseModel):
    left: ColumnRef
    right: ColumnRef
    score: float
    band: ConfidenceBand
    evidence: MatchEvidence
    relationship: str = "similar_column"  # similar_column | foreign_key_candidate | semantic_match
    accepted: bool = False  # survived bipartite alignment
    status: str = "proposed"  # proposed | approved | rejected
    rejection_reason: str | None = None

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.left.key, self.right.key)


class JoinKind(str, Enum):
    LOOKUP = "lookup"  # many-to-one onto a unique key
    ENTITY_KEY_MERGE = "entity_key_merge"  # same-grain tables sharing an id space
    ENTITY_RESOLUTION = "entity_resolution"  # same entities, no reliable shared key
    AGGREGATE_LOOKUP = "aggregate_lookup"  # key not unique on the dimension side
    NONE = "none"


class DatasetRelationship(BaseModel):
    left_dataset: str
    right_dataset: str
    confidence: float
    band: ConfidenceBand
    join_kind: JoinKind
    key_matches: list[ColumnMatch] = Field(default_factory=list)
    attribute_matches: list[ColumnMatch] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""


# --------------------------------------------------------------------------- entity resolution


class EntityPair(BaseModel):
    left_dataset: str
    left_row: int
    right_dataset: str
    right_row: int
    probability: float
    match_weight: float
    field_levels: dict[str, str] = Field(default_factory=dict)
    field_scores: dict[str, float] = Field(default_factory=dict)


class EntityCluster(BaseModel):
    entity_id: str
    members: list[tuple[str, int]]  # (dataset_id, row index)
    confidence: float
    size: int


# --------------------------------------------------------------------------- planning & merge


class MergeStep(BaseModel):
    step: int
    operation: str
    description: str
    left: str | None = None
    right: str | None = None
    join_kind: JoinKind = JoinKind.NONE
    join_type: str | None = None  # left | full_outer | inner
    join_keys: list[dict[str, str]] = Field(default_factory=list)
    role: str | None = None
    transformations: list[str] = Field(default_factory=list)
    confidence: float | None = None
    threshold: float | None = None
    requires_review: bool = False


class CanonicalColumn(BaseModel):
    name: str
    semantic_type: SemanticType
    data_type: DataType
    sources: list[ColumnRef]
    role: str | None = None
    transformation: str | None = None


class MergePlan(BaseModel):
    plan_id: str
    mode: str
    conflict_strategy: str
    thresholds: dict[str, float]
    root_dataset: str
    grain: str
    datasets: list[str]
    steps: list[MergeStep]
    canonical_schema: list[CanonicalColumn]
    excluded_relationships: list[dict[str, Any]] = Field(default_factory=list)
    integration_tree: list[dict[str, Any]] = Field(default_factory=list)
    entity_groups: list[dict[str, Any]] = Field(default_factory=list)
    unmerged_datasets: list[str] = Field(default_factory=list)
    routes: list[dict[str, Any]] = Field(default_factory=list)
    transformations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class Conflict(BaseModel):
    entity_id: str
    attribute: str
    candidates: list[dict[str, Any]]  # {value, source_dataset, source_column, source_row, confidence}
    resolved_value: Any = None
    strategy: str
    reason: str
    status: str = "flagged"  # flagged | resolved | manual_review


class ProvenanceRecord(BaseModel):
    output_column: str
    source_dataset: str
    source_column: str
    transformation: str | None = None
    merge_operation: str
    confidence: float | None = None
    timestamp: datetime = Field(default_factory=utcnow)


class MergeResult(BaseModel):
    merge_id: str
    plan_id: str
    output_path: str
    row_count: int
    column_count: int
    columns: list[str]
    matched_rows: int
    unmatched_rows: int
    duplicates_resolved: int
    conflicts: int
    entity_clusters: int
    elapsed_ms: float
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
