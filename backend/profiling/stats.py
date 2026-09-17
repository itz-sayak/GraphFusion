"""DuckDB-backed column statistics.

All statistics are computed by SQL aggregates that DuckDB evaluates directly
over the Parquet file with a streaming, vectorised scan. Python never holds
more than the per-column bottom-k value sample (default k = 20 000) and the
top-value list, so profiling memory is O(columns · k), independent of rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.models import DataType
from backend.storage.duck import connect, parquet_scan, quote_ident

EXACT_DISTINCT_MAX_ROWS = 2_000_000

# SQL expression equivalent to backend.core.values.basic_normalize for sketching
_NORM_SQL = (
    "regexp_replace(strip_accents(lower(trim(regexp_replace(CAST({col} AS VARCHAR), '\\s+', ' ', 'g')))), "
    "'^(-?[0-9]+)\\.0+$', '\\1')"
)


def physical_to_datatype(physical: str) -> DataType:
    p = physical.upper()
    if any(t in p for t in ("INT", "HUGEINT")):
        return DataType.INTEGER
    if any(t in p for t in ("DOUBLE", "FLOAT", "DECIMAL", "REAL")):
        return DataType.FLOAT
    if "BOOL" in p:
        return DataType.BOOLEAN
    if "TIMESTAMP" in p:
        return DataType.DATETIME
    if p == "DATE":
        return DataType.DATE
    if "VARCHAR" in p or "STRING" in p or "TEXT" in p:
        return DataType.STRING
    return DataType.OTHER


def table_stats(path: str | Path, bottom_k: int = 20000, top_n: int = 25) -> dict[str, Any]:
    src = parquet_scan(path)
    with connect() as con:
        con.execute(f"CREATE VIEW t AS SELECT * FROM {src}")
        described = con.execute("DESCRIBE t").fetchall()
        columns = [(r[0], r[1]) for r in described]
        row_count = con.execute("SELECT count(*) FROM t").fetchone()[0]
        exact = row_count <= EXACT_DISTINCT_MAX_ROWS

        dup_rows = 0
        if columns and row_count:
            dup_rows = con.execute("SELECT (SELECT count(*) FROM t) - (SELECT count(*) FROM (SELECT DISTINCT * FROM t))").fetchone()[0]

        # one scan for the basic aggregates of every column
        agg_parts: list[str] = []
        for name, physical in columns:
            q = quote_ident(name)
            dt = physical_to_datatype(physical)
            distinct = f"count(DISTINCT {q})" if exact else f"approx_count_distinct({q})"
            agg_parts += [f"count({q})", distinct]
            if dt == DataType.STRING:
                agg_parts += [
                    f"avg(CASE WHEN TRY_CAST({q} AS DOUBLE) IS NOT NULL THEN 1.0 ELSE 0.0 END) FILTER (WHERE {q} IS NOT NULL)",
                    f"avg(length({q}))",
                ]
            else:
                agg_parts += ["NULL", "NULL"]
        agg = con.execute(f"SELECT {', '.join(agg_parts)} FROM t").fetchone() if agg_parts else ()

        results: dict[str, dict[str, Any]] = {}
        for i, (name, physical) in enumerate(columns):
            q = quote_ident(name)
            non_null, distinct, numeric_rate, avg_len = agg[i * 4 : i * 4 + 4]
            dt = physical_to_datatype(physical)
            numeric_expr = None
            if dt in (DataType.INTEGER, DataType.FLOAT):
                numeric_expr = f"CAST({q} AS DOUBLE)"
            elif dt == DataType.STRING and numeric_rate is not None and numeric_rate >= 0.98 and non_null:
                numeric_expr = f"TRY_CAST({q} AS DOUBLE)"
            elif dt == DataType.BOOLEAN:
                numeric_expr = f"CAST({q} AS DOUBLE)"

            col: dict[str, Any] = {
                "name": name,
                "physical_type": physical,
                "data_type": dt,
                "non_null": int(non_null or 0),
                "null_count": int(row_count - (non_null or 0)),
                "distinct": int(distinct or 0),
                "numeric_string_rate": float(numeric_rate) if numeric_rate is not None else None,
                "avg_length": float(avg_len) if avg_len is not None else None,
            }
            if numeric_expr and non_null:
                mn, mx, mean, std, q05, q25, q50, q75, q95, integral = con.execute(
                    f"SELECT min(x), max(x), avg(x), stddev_samp(x), quantile_cont(x, 0.05), quantile_cont(x, 0.25), "
                    f"quantile_cont(x, 0.5), quantile_cont(x, 0.75), quantile_cont(x, 0.95), "
                    f"avg(CASE WHEN x = floor(x) THEN 1.0 ELSE 0.0 END) FROM (SELECT {numeric_expr} AS x FROM t) WHERE x IS NOT NULL"
                ).fetchone()
                col.update(min=mn, max=mx, mean=mean, std=std, quantiles=[q05, q25, q50, q75, q95], integral_rate=integral)
                if dt == DataType.STRING:
                    col["data_type"] = DataType.INTEGER if (integral or 0) >= 0.999 else DataType.FLOAT
                    col["numeric_from_string"] = True
            elif non_null and dt in (DataType.STRING, DataType.DATE, DataType.DATETIME):
                mn, mx = con.execute(f"SELECT CAST(min({q}) AS VARCHAR), CAST(max({q}) AS VARCHAR) FROM t").fetchone()
                col.update(min=mn, max=mx)

            if non_null:
                col["top_values"] = [
                    (v, int(c))
                    for v, c in con.execute(
                        f"SELECT CAST({q} AS VARCHAR) AS v, count(*) AS c FROM t WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY c DESC, v LIMIT {top_n}"
                    ).fetchall()
                ]
                norm = _NORM_SQL.format(col=q)
                sample = con.execute(
                    f"SELECT v, hash(v) AS h FROM (SELECT DISTINCT {norm} AS v FROM t WHERE {q} IS NOT NULL) "
                    f"WHERE v IS NOT NULL AND v <> '' ORDER BY h LIMIT {bottom_k + 1}"
                ).fetchall()
                complete = len(sample) <= bottom_k
                sample = sample[:bottom_k]
                col["sketch_values"] = {v: int(h) for v, h in sample}
                col["sketch_complete"] = complete
                col["sketch_tau"] = int(sample[-1][1]) if sample else 0
                col["sample_values"] = [
                    r[0] for r in con.execute(f"SELECT DISTINCT CAST({q} AS VARCHAR) FROM t WHERE {q} IS NOT NULL LIMIT 10").fetchall()
                ]
                col["raw_sample"] = [
                    r[0]
                    for r in con.execute(
                        f"SELECT CAST({q} AS VARCHAR) FROM t WHERE {q} IS NOT NULL USING SAMPLE reservoir(1000 ROWS) REPEATABLE (42)"
                    ).fetchall()
                ]
            else:
                col.update(top_values=[], sketch_values={}, sketch_complete=True, sketch_tau=0, sample_values=[], raw_sample=[])
            results[name] = col

    return {"row_count": int(row_count), "duplicate_rows": int(dup_rows), "columns": results, "order": [c[0] for c in columns]}
