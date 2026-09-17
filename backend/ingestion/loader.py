"""Build :class:`DatasetArtifact` objects from arbitrary sources.

Guarantees:
* the original file is never opened for writing; a byte-identical read-only
  copy is kept in ``raw/`` with its SHA-256 so every result is reproducible;
* all downstream stages read only the normalised Parquet in ``processed/``.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from backend.core.models import DatasetArtifact, SourceType
from backend.ingestion.detect import detect_format
from backend.ingestion.readers import fetch_rest, write_parquet
from backend.logging_conf import get_logger
from backend.storage.workspace import Workspace

log = get_logger(__name__)


def make_dataset_id(source_name: str, existing: set[str] | None = None) -> str:
    stem = Path(source_name).stem if "://" not in source_name else source_name.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "dataset"
    candidate, i = slug, 2
    while existing and candidate in existing:
        candidate, i = f"{slug}_{i}", i + 1
    return candidate


def _parquet_schema(path: Path) -> tuple[dict[str, str], int]:
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    return {f.name: str(f.type) for f in schema}, meta.num_rows


def ingest_file(
    workspace: Workspace,
    path: str | Path,
    source_name: str | None = None,
    existing_ids: set[str] | None = None,
    options: dict[str, Any] | None = None,
) -> DatasetArtifact:
    started = time.perf_counter()
    path = Path(path)
    source_name = source_name or path.name
    source_type = detect_format(path)
    dataset_id = make_dataset_id(source_name, existing_ids)

    raw_copy, checksum = workspace.preserve_original(path, dataset_id)
    dest = workspace.processed_path(dataset_id)
    reader_meta = write_parquet(source_type, raw_copy, dest, options)
    schema, rows = _parquet_schema(dest)

    artifact = DatasetArtifact(
        dataset_id=dataset_id,
        source_name=source_name,
        source_type=source_type,
        source_uri=str(path.resolve()),
        storage_path=str(dest),
        schema=schema,
        row_count=rows,
        column_count=len(schema),
        checksum=checksum,
        metadata={**reader_meta, "raw_copy": str(raw_copy), "bytes": raw_copy.stat().st_size},
    )
    log.info(
        "Dataset ingested",
        dataset_id=dataset_id,
        format=source_type.value,
        rows=rows,
        columns=len(schema),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return artifact


def ingest_rest(
    workspace: Workspace,
    url: str,
    source_name: str,
    existing_ids: set[str] | None = None,
    params: dict[str, Any] | None = None,
    page_size: int | None = None,
) -> DatasetArtifact:
    """OPTIONAL source: ingest a JSON REST endpoint."""
    dataset_id = make_dataset_id(source_name, existing_ids)
    dest = workspace.processed_path(dataset_id)
    meta = fetch_rest(url, dest, params=params, page_size=page_size)
    schema, rows = _parquet_schema(dest)
    log.info("Dataset ingested", dataset_id=dataset_id, format="rest", rows=rows, columns=len(schema))
    return DatasetArtifact(
        dataset_id=dataset_id,
        source_name=source_name,
        source_type=SourceType.REST,
        source_uri=url,
        storage_path=str(dest),
        schema=schema,
        row_count=rows,
        column_count=len(schema),
        metadata=meta,
    )
