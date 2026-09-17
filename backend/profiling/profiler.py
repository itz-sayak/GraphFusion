"""Dataset profiling orchestration: statistics → semantic types → sketches."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any

from backend.core.models import ColumnProfile, DataType, DatasetArtifact, DatasetProfile
from backend.core.values import detect_date_formats, is_dayfirst, value_shape
from backend.logging_conf import get_logger
from backend.profiling.semantic_types import infer_semantic_type
from backend.profiling.sketches import ColumnSketch, build_minhash
from backend.profiling.stats import table_stats

log = get_logger(__name__)


def cardinality_label(distinct: int, non_null: int) -> str:
    if non_null == 0:
        return "empty"
    if distinct <= 1:
        return "constant"
    ratio = distinct / non_null
    if ratio >= 0.98:
        return "unique"
    if ratio >= 0.5:
        return "high"
    if distinct <= 60 or ratio < 0.05:
        return "low"
    return "medium"


def pattern_histogram(sample: list[Any], top: int = 8) -> dict[str, float]:
    if not sample:
        return {}
    counts = Counter(value_shape(v) for v in sample)
    total = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.most_common(top)}


def profile_dataset(artifact: DatasetArtifact, config: dict[str, Any]) -> tuple[DatasetProfile, dict[str, ColumnSketch]]:
    started = time.perf_counter()
    sm = config["schema_matching"]
    raw = table_stats(artifact.storage_path, bottom_k=sm["max_distinct_sample"])
    row_count = raw["row_count"]
    profiles: list[ColumnProfile] = []
    sketches: dict[str, ColumnSketch] = {}

    for name in raw["order"]:
        st = raw["columns"][name]
        sem, sem_conf, sem_evidence = infer_semantic_type(name, st, row_count)
        date_format = None
        dayfirst = None
        if st["data_type"] == DataType.STRING and sem.value in ("DATE", "DATETIME"):
            _, date_format, _ = detect_date_formats(st["raw_sample"][:500])
            dayfirst = is_dayfirst(st["raw_sample"])
        non_null = st["non_null"]
        profiles.append(
            ColumnProfile(
                name=name,
                physical_type=st["physical_type"],
                data_type=st["data_type"],
                semantic_type=sem,
                semantic_confidence=sem_conf,
                semantic_evidence=sem_evidence,
                null_count=st["null_count"],
                null_pct=round(st["null_count"] / row_count, 4) if row_count else 0.0,
                unique_count=st["distinct"],
                uniqueness=round(st["distinct"] / non_null, 4) if non_null else 0.0,
                cardinality=cardinality_label(st["distinct"], non_null),
                min=_jsonable(st.get("min")),
                max=_jsonable(st.get("max")),
                mean=st.get("mean"),
                std=st.get("std"),
                quantiles=st.get("quantiles"),
                avg_length=st.get("avg_length"),
                sample_values=st["sample_values"],
                top_values=st["top_values"],
                pattern_histogram=pattern_histogram(st["raw_sample"]),
                date_format=date_format,
                date_dayfirst=dayfirst,
            )
        )
        sketch = ColumnSketch(
            column=name,
            values=st["sketch_values"],
            complete=st["sketch_complete"],
            tau=st["sketch_tau"],
            distinct_estimate=st["distinct"],
        )
        sketch.minhash = build_minhash(sketch.values, sm["minhash_permutations"])
        sketches[name] = sketch

    elapsed = (time.perf_counter() - started) * 1000
    profile = DatasetProfile(
        dataset_id=artifact.dataset_id,
        row_count=row_count,
        column_count=len(profiles),
        duplicate_rows=raw["duplicate_rows"],
        columns=profiles,
        elapsed_ms=round(elapsed, 1),
    )
    log.info("Schema profiling completed", dataset_id=artifact.dataset_id, columns=len(profiles), rows=row_count, elapsed_ms=round(elapsed, 1))
    return profile, sketches


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return str(v)
