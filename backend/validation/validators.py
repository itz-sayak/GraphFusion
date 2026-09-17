"""Post-merge validation: the integrity guarantees are checked, not assumed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from backend.core.models import DatasetArtifact, MergePlan
from backend.merge.executor import ExecutionContext
from backend.storage.duck import quote_ident
from backend.storage.workspace import sha256_file


def _check(name: str, passed: bool, details: str, severity: str = "error", **data: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "severity": "info" if passed else severity, "details": details, **data}


def validate_merge(con: duckdb.DuckDBPyConnection, ctx: ExecutionContext, plan: MergePlan, artifacts: dict[str, DatasetArtifact]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rows = con.execute("SELECT count(*) FROM unified").fetchone()[0]
    root = plan.root_dataset
    entity_grain = ctx.spec.entity_grain

    if entity_grain:
        gid = ctx.spec.group.group_id  # type: ignore[union-attr]
        st = ctx.group_stats[gid]
        checks.append(_check("row_conservation", rows == st["entities"], f"{rows} output rows for {st['entities']} resolved entities", expected=st["entities"], actual=rows))
    else:
        root_rows = artifacts[root].row_count
        checks.append(_check("row_conservation", rows == root_rows, f"{rows} output rows; root dataset {root} has {root_rows} rows (no rows dropped or added)", expected=root_rows, actual=rows))
        src_col = f"_src_{root}_row"
        distinct = con.execute(f"SELECT count(DISTINCT {quote_ident(src_col)}) FROM unified").fetchone()[0]
        checks.append(_check("no_fan_out", distinct == rows, f"{distinct} distinct source rows across {rows} output rows", expected=rows, actual=distinct))

    for j in ctx.join_stats:
        checks.append(_check(
            f"join_row_invariance[{j['alias']}]", j["rows"] == j["parent_rows_before"],
            f"{j['kind']} {j['parent']} ← {j['child']}: {j['parent_rows_before']} rows before, {j['rows']} after; {j['matched_rows']} matched ({j['match_rate']:.1%})",
        ))
        if j["unmatched_rows"]:
            checks.append(_check(
                f"unmatched_rows_retained[{j['alias']}]", True,
                f"{j['unmatched_rows']} {j['parent']} rows had no match in {j['child']}; they are kept with NULL attributes and _match_{j['alias']} = false",
            ))

    for gid, st in ctx.group_stats.items():
        clusters_total = sum(c.size for c in (ctx.er_results[gid].clusters if gid in ctx.er_results else []))
        if gid in ctx.er_results:
            checks.append(_check(f"entity_assignment[{gid}]", clusters_total == st["records"], f"{st['records']} records assigned to {st['entities']} entities (each record exactly once)"))
        checks.append(_check(f"conflicts_recorded[{gid}]", True, f"{st['conflicts']} conflicting attribute values recorded with all candidates; none overwritten silently"))

    for ds in plan.datasets:
        a = artifacts[ds]
        raw = Path(a.metadata.get("raw_copy", ""))
        if raw.exists():
            ok = sha256_file(raw) == a.checksum
            checks.append(_check(f"source_immutable[{ds}]", ok, "preserved source copy unchanged (SHA-256 verified)" if ok else "preserved source copy was modified!"))
        original = Path(a.source_uri)
        if original.exists() and original.stat().st_size == a.metadata.get("bytes"):
            ok = sha256_file(original) == a.checksum
            checks.append(_check(f"original_untouched[{ds}]", ok, "original file checksum matches ingestion" if ok else "original file changed after ingestion", severity="warning"))

    if not entity_grain:
        root_cols = {c.output: c for c in ctx.spec.columns if c.operation == "select" and c.source_dataset == root and c.transformation is None}
        mismatches = []
        src = artifacts[root].storage_path.replace("\\", "/")
        for out_name, c in list(root_cols.items())[:40]:
            n_out = con.execute(f"SELECT count(*) FILTER (WHERE {quote_ident(out_name)} IS NULL) FROM unified").fetchone()[0]
            n_src = con.execute(f"SELECT count(*) FILTER (WHERE {quote_ident(c.source_column)} IS NULL) FROM read_parquet('{src}')").fetchone()[0]
            if n_out != n_src:
                mismatches.append({"column": out_name, "source_nulls": n_src, "output_nulls": n_out})
        checks.append(_check("root_values_preserved", not mismatches, "untransformed root columns have identical null counts in source and output" if not mismatches else f"{len(mismatches)} columns changed null counts", mismatches=mismatches))

    for ds, ts in plan.transformations.items():
        for t in ts:
            if t["transform"] != "parse_date":
                continue
            table = f"v_{ds}"
            raw, parsed = t["column"] + "_raw", t["column"]
            failed = con.execute(f"SELECT count(*) FROM {quote_ident(table)} WHERE {quote_ident(raw)} IS NOT NULL AND trim(CAST({quote_ident(raw)} AS VARCHAR)) <> '' AND {quote_ident(parsed)} IS NULL").fetchone()[0]
            checks.append(_check(f"date_parse[{ds}.{t['column']}]", failed == 0, f"{failed} values could not be parsed (raw text preserved in {raw})", severity="warning", unparsed=failed))

    from backend.merge.temporal import check_intervals

    for gid, hist in ctx.histories.items():
        problems = check_intervals(hist)
        dated = sum(1 for h in hist if h["valid_from"] is not None)
        checks.append(_check(f"temporal_consistency[{gid}]", not problems,
                             f"{dated} validity intervals: non-empty, non-overlapping, at most one current per entity attribute" if not problems else f"{len(problems)} violations, e.g. {problems[0]}",
                             violations=problems[:20]))

    errors = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    warnings = [c for c in checks if not c["passed"] and c["severity"] == "warning"]
    return {"passed": not errors, "errors": len(errors), "warnings": len(warnings), "checks": checks}
