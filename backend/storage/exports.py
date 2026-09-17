"""Export bundle writer.

    unified_dataset.parquet / .csv   the integrated dataset (with _record_id, _src_*, _match_* lineage columns)
    merge_report.json                plan, execution statistics, validation, quality report
    provenance.json                  column lineage + W3C PROV-JSON document
    conflicts.csv                    one row per conflict candidate (value, source, selected?)
    schema_mapping.json              every scored correspondence with its evidence and decision
    integration_graph.json           nodes/edges with confidence, cost and evidence
    entities_<group>.parquet         full-outer entity table per entity group (incl. entities not referenced by the root)
    cell_provenance.parquet          which record supplied each fused value
    data_quality_report.txt          human-readable quality report
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from backend.core.models import Conflict
from backend.storage.duck import quote_ident, quote_literal


def _dump(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def write_conflicts_csv(path: Path, conflicts: list[Conflict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entity_id", "attribute", "status", "strategy", "resolved_value", "reason", "candidate_value", "source_dataset", "source_column", "source_row", "confidence", "record_timestamp", "selected"])
        for c in conflicts:
            for cand in c.candidates:
                w.writerow([c.entity_id, c.attribute, c.status, c.strategy, c.resolved_value, c.reason, cand["value"], cand["source_dataset"], cand["source_column"], cand["source_row"], cand["confidence"], cand["record_timestamp"], cand["selected"]])


def write_bundle(
    out_dir: Path,
    con: duckdb.DuckDBPyConnection,
    entity_tables: dict[str, str],
    merge_report: dict[str, Any],
    provenance: dict[str, Any],
    conflicts: list[Conflict],
    cell_provenance: list[dict[str, Any]],
    schema_mapping: dict[str, Any],
    graph_json: dict[str, Any],
    quality_text: str,
    write_csv: bool = True,
    histories: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    parquet = out_dir / "unified_dataset.parquet"
    if not parquet.exists():
        con.execute(f"COPY unified TO {quote_literal(parquet.as_posix())} (FORMAT parquet, COMPRESSION zstd)")
    files["unified_dataset.parquet"] = str(parquet)
    if write_csv:
        csv_path = out_dir / "unified_dataset.csv"
        con.execute(f"COPY unified TO {quote_literal(csv_path.as_posix())} (HEADER, DELIMITER ',')")
        files["unified_dataset.csv"] = str(csv_path)
    for gid, table in entity_tables.items():
        p = out_dir / f"entities_{gid}.parquet"
        cols = [r[0] for r in con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()]
        names = merge_report.get("entity_table_columns", {}).get(gid, {})
        select = ", ".join(f"{quote_ident(c)} AS {quote_ident(names.get(c, c))}" for c in cols if c != "__entity_id" and names.get(c))
        con.execute(f"COPY (SELECT {select or '*'} FROM {quote_ident(table)}) TO {quote_literal(p.as_posix())} (FORMAT parquet)")
        files[p.name] = str(p)
    for gid, hist in (histories or {}).items():
        p = out_dir / f"entity_history_{gid}.parquet"
        pq.write_table(pa.table({
            "entity_id": pa.array([h["entity_id"] for h in hist], pa.string()),
            "attribute": pa.array([h["attribute"] for h in hist], pa.string()),
            "value": pa.array([None if h["value"] is None else str(h["value"]) for h in hist], pa.string()),
            "valid_from": pa.array([h["valid_from"] for h in hist], pa.timestamp("us")),
            "valid_to": pa.array([h["valid_to"] for h in hist], pa.timestamp("us")),
            "is_current": pa.array([h["is_current"] for h in hist], pa.bool_()),
            "timestamp_kind": pa.array([h["timestamp_kind"] for h in hist], pa.string()),
            "supporting_records": pa.array([h["supporting_records"] for h in hist], pa.int32()),
            "source_dataset": pa.array([h["source_dataset"] for h in hist], pa.string()),
            "source_column": pa.array([h["source_column"] for h in hist], pa.string()),
            "source_row": pa.array([h["source_row"] for h in hist], pa.int64()),
            # transaction time: when this merge recorded the interval (valid time is valid_from / valid_to)
            "recorded_at": pa.array([h.get("recorded_at") for h in hist], pa.timestamp("us")),
        }), p)
        files[p.name] = str(p)
    if cell_provenance:
        p = out_dir / "cell_provenance.parquet"
        keys = list(cell_provenance[0])
        pq.write_table(pa.table({k: pa.array([str(r[k]) if k in ("source_row",) and r[k] is not None else r[k] for r in cell_provenance]) for k in keys}), p)
        files[p.name] = str(p)
    for name, obj in (("merge_report.json", merge_report), ("provenance.json", provenance), ("schema_mapping.json", schema_mapping), ("integration_graph.json", graph_json)):
        _dump(out_dir / name, obj)
        files[name] = str(out_dir / name)
    write_conflicts_csv(out_dir / "conflicts.csv", conflicts)
    files["conflicts.csv"] = str(out_dir / "conflicts.csv")
    (out_dir / "data_quality_report.txt").write_text(quality_text, encoding="utf-8")
    files["data_quality_report.txt"] = str(out_dir / "data_quality_report.txt")
    return files
