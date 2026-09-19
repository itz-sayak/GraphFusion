"""Format-specific readers that write a normalised Parquet file.

Every reader streams into Parquet through DuckDB where the format allows it
(CSV, JSONL, Parquet, large JSON), so ingesting a multi-GB file never
materialises it in Python memory. Formats that are inherently small or not
streamable (XLSX, SQLite tables, REST payloads, irregular JSON) go through
PyArrow. Nested JSON structures are flattened to dotted column names.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.core.errors import IngestionError, UnsupportedFormatError
from backend.core.models import SourceType
from backend.storage.duck import connect, quote_ident, quote_literal

LARGE_JSON_BYTES = 256 * 1024 * 1024
NULL_SQL = r"'', 'NULL', '\N'"  # empty, and the null markers of SQL exports


def write_parquet(source_type: SourceType, src: Path, dest: Path, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert ``src`` into Parquet at ``dest``. Returns reader metadata."""
    options = options or {}
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_type == SourceType.PARQUET:
            shutil.copyfile(src, dest)
            return {"reader": "parquet-copy"}
        if source_type == SourceType.CSV:
            return _csv(src, dest, options)
        if source_type == SourceType.JSONL:
            return _duck_json(src, dest, "newline_delimited")
        if source_type == SourceType.JSON:
            if src.stat().st_size > LARGE_JSON_BYTES:
                return _duck_json(src, dest, "array")
            with open(src, encoding="utf-8-sig") as fh:
                payload = json.load(fh)
            return records_to_parquet(payload, dest)
        if source_type == SourceType.XLSX:
            return _xlsx(src, dest, options)
        if source_type == SourceType.SQLITE:
            return _sqlite(src, dest, options)
    except (UnsupportedFormatError, IngestionError):
        raise
    except Exception as exc:  # noqa: BLE001 - surface any reader failure uniformly
        raise IngestionError(f"Failed to read {src.name} as {source_type.value}: {exc}") from exc
    raise UnsupportedFormatError(f"No reader for {source_type.value}")


# --------------------------------------------------------------------------- CSV / JSONL


def _csv(src: Path, dest: Path, options: dict[str, Any]) -> dict[str, Any]:
    """Lossless CSV ingestion.

    Sample-based sniffing picks a type from the first N rows and then fails
    (or silently coerces) on later rows — fatal for mixed date formats and
    for identifiers with leading zeros. Instead we read every column as text
    and promote a column to BIGINT / DOUBLE / BOOLEAN / TIMESTAMP only when
    *every* non-null value casts, and no value has a significant leading zero.
    Two streaming passes; memory stays bounded.
    """
    delim = options.get("delimiter")
    args = [quote_literal(src.as_posix()), "header=true", "all_varchar=true"]
    if delim:
        args.append(f"delim={quote_literal(delim)}")
    reader = f"read_csv({', '.join(args)})"
    promotions: dict[str, str] = {}
    with connect() as con:
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {reader}").fetchall()]
    if len(cols) == 1 and not delim and not options.get("_repaired"):
        # the sniffer collapsed a delimited file into one column: usually ragged rows (a delimiter inside an
        # unquoted text value). Rebuild a well-formed file and read that instead.
        from backend.ingestion.csv_repair import needs_repair, repair

        d = needs_repair(src)
        if d:
            fixed = dest.with_name(dest.stem + ".repaired.csv")
            try:
                stats = repair(src, fixed, d)
                meta = _csv(fixed, dest, {**options, "delimiter": ",", "_repaired": True})
            finally:
                fixed.unlink(missing_ok=True)
            return {**meta, "csv_repair": stats}
    with connect() as con:
        if cols:
            checks: list[str] = []
            for c in cols:
                q = quote_ident(c)
                nn = f"{q} IS NOT NULL AND trim({q}) NOT IN ({NULL_SQL})"
                checks += [
                    # TRY_CAST('11.5' AS BIGINT) rounds instead of failing, so require an integer literal
                    f"bool_and(regexp_matches(trim({q}), '^-?[0-9]+$') AND TRY_CAST({q} AS BIGINT) IS NOT NULL AND NOT regexp_matches(trim({q}), '^-?0[0-9]')) FILTER (WHERE {nn})",
                    f"bool_and(TRY_CAST({q} AS DOUBLE) IS NOT NULL AND NOT regexp_matches(trim({q}), '^-?0[0-9]')) FILTER (WHERE {nn})",
                    f"bool_and(lower(trim({q})) IN ('true','false')) FILTER (WHERE {nn})",
                    f"bool_and(regexp_matches(trim({q}), '^\\d{{4}}-\\d{{2}}-\\d{{2}}([ T]\\d{{2}}:\\d{{2}}(:\\d{{2}}(\\.\\d+)?)?)?$') AND TRY_CAST({q} AS TIMESTAMP) IS NOT NULL) FILTER (WHERE {nn})",
                ]
            flags = con.execute(f"SELECT {', '.join(checks)} FROM {reader}").fetchone()
            for i, c in enumerate(cols):
                is_int, is_float, is_bool, is_ts = flags[i * 4 : i * 4 + 4]
                if is_int:
                    promotions[c] = "BIGINT"
                elif is_float:
                    promotions[c] = "DOUBLE"
                elif is_bool:
                    promotions[c] = "BOOLEAN"
                elif is_ts:
                    promotions[c] = "TIMESTAMP"
        select = []
        for c in cols:
            q = quote_ident(c)
            if c in promotions:
                select.append(f"CAST(CASE WHEN trim({q}) IN ({NULL_SQL}) THEN NULL ELSE trim({q}) END AS {promotions[c]}) AS {q}")
            else:  # SQL-export null markers (NULL, \N) are missing values; other text is kept as written
                select.append(f"CASE WHEN {q} IN ({NULL_SQL}) THEN NULL ELSE {q} END AS {q}")
        con.execute(f"COPY (SELECT {', '.join(select) or '*'} FROM {reader}) TO {quote_literal(dest.as_posix())} (FORMAT parquet)")
    return {"reader": "duckdb.read_csv(lossless)", "promoted_types": promotions}


def _duck_json(src: Path, dest: Path, fmt: str) -> dict[str, Any]:
    with connect() as con:
        con.execute(
            f"COPY (SELECT * FROM read_json_auto({quote_literal(src.as_posix())}, format={quote_literal(fmt)}, sample_size=20000)) "
            f"TO {quote_literal(dest.as_posix())} (FORMAT parquet)"
        )
    _flatten_parquet_structs(dest)
    return {"reader": f"duckdb.read_json({fmt})"}


def _flatten_parquet_structs(path: Path) -> None:
    """Flatten struct columns (``a.b``) in place using DuckDB's struct access."""
    schema = pq.read_schema(path)
    if not any(pa.types.is_struct(f.type) for f in schema):
        return
    exprs: list[str] = []

    def walk(field: pa.Field, prefix_sql: str, prefix_name: str) -> None:
        if pa.types.is_struct(field.type):
            for child in field.type:
                walk(child, f"{prefix_sql}[{quote_literal(child.name)}]", f"{prefix_name}.{child.name}")
        else:
            exprs.append(f"{prefix_sql} AS {quote_ident(prefix_name)}")

    for f in schema:
        walk(f, quote_ident(f.name), f.name)
    tmp = path.with_suffix(".flat.parquet")
    with connect() as con:
        con.execute(
            f"COPY (SELECT {', '.join(exprs)} FROM read_parquet({quote_literal(path.as_posix())})) TO {quote_literal(tmp.as_posix())} (FORMAT parquet)"
        )
    tmp.replace(path)


# --------------------------------------------------------------------------- JSON records


def extract_records(payload: Any) -> list[dict[str, Any]]:
    """Find the record list inside common JSON envelopes.

    Supports: a list of objects; a list of lists whose first row is a header
    (Census API format); an object wrapping a list under any key
    (``{"data": [...]}``, ``{"results": [...]}``); and a single object.
    """
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(r, list) for r in payload) and all(isinstance(h, str) for h in payload[0]):
            header = payload[0]
            return [dict(zip(header, row)) for row in payload[1:]]
        if all(isinstance(r, dict) for r in payload):
            return payload
        raise IngestionError("JSON array must contain objects or a header row followed by rows")
    if isinstance(payload, dict):
        list_values = [(k, v) for k, v in payload.items() if isinstance(v, list) and v and isinstance(v[0], (dict, list))]
        if list_values:
            key, value = max(list_values, key=lambda kv: len(kv[1]))
            return extract_records(value)
        return [payload]
    raise IngestionError("Unsupported JSON document structure")


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_record(value, name))
        elif isinstance(value, list):
            out[name] = json.dumps(value, ensure_ascii=False)
        else:
            out[name] = value
    return out


def records_to_parquet(payload: Any, dest: Path) -> dict[str, Any]:
    records = [flatten_record(r) for r in extract_records(payload)]
    if not records:
        raise IngestionError("JSON source contains no records")
    columns: dict[str, list[Any]] = {}
    for r in records:
        for k in r:
            columns.setdefault(k, [])
    for r in records:
        for k, col in columns.items():
            col.append(r.get(k))
    arrays = {k: _to_arrow_array(v) for k, v in columns.items()}
    pq.write_table(pa.table(arrays), dest)
    return {"reader": "json-records", "records": len(records)}


def _to_arrow_array(values: list[Any]) -> pa.Array:
    try:
        arr = pa.array(values)
        if pa.types.is_null(arr.type):
            return pa.array(values, type=pa.string())
        return arr
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.array([None if v is None else str(v) for v in values], type=pa.string())


# --------------------------------------------------------------------------- XLSX / SQLite


def _xlsx(src: Path, dest: Path, options: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
    sheet = options.get("sheet")
    if sheet is None:
        sheet = next((name for name, df in sheets.items() if not df.empty), None)
    if sheet is None or sheet not in sheets:
        raise IngestionError(f"Sheet not found in {src.name}: {sheet}")
    df = sheets[sheet]
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v) if not isinstance(v, (int, float, bool)) else v)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, dest)
    return {"reader": "openpyxl", "sheet": sheet, "sheets": list(sheets)}


def list_sqlite_tables(src: Path) -> list[str]:
    with sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True) as con:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


def _sqlite(src: Path, dest: Path, options: dict[str, Any]) -> dict[str, Any]:
    tables = list_sqlite_tables(src)
    table = options.get("table") or (tables[0] if tables else None)
    if table not in tables:
        raise IngestionError(f"Table {table!r} not found in {src.name}; available: {tables}")
    with sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True) as con:
        cur = con.execute(f"SELECT * FROM {quote_ident(table)}")
        names = [d[0] for d in cur.description]
        writer = None
        try:
            while rows := cur.fetchmany(50_000):
                cols = list(zip(*rows))
                batch = pa.table({n: _to_arrow_array(list(c)) for n, c in zip(names, cols)})
                if writer is None:
                    writer = pq.ParquetWriter(dest, batch.schema)
                writer.write_table(batch.cast(writer.schema, safe=False))
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            pq.write_table(pa.table({n: pa.array([], type=pa.string()) for n in names}), dest)
    return {"reader": "sqlite3", "table": table, "tables": tables}


# --------------------------------------------------------------------------- optional sources


def fetch_rest(url: str, dest: Path, params: dict[str, Any] | None = None, page_size: int | None = None, max_pages: int = 100) -> dict[str, Any]:
    """OPTIONAL: pull a JSON REST endpoint (Socrata-style ``$limit/$offset`` paging supported)."""
    import httpx

    records: list[dict[str, Any]] = []
    params = dict(params or {})
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for page in range(max_pages):
            if page_size:
                params.update({"$limit": page_size, "$offset": page * page_size})
            resp = client.get(url, params=params)
            resp.raise_for_status()
            batch = extract_records(resp.json())
            records.extend(batch)
            if not page_size or len(batch) < page_size:
                break
    return {**records_to_parquet(records, dest), "reader": "rest", "url": url}


def fetch_postgres(dsn: str, query: str, dest: Path) -> dict[str, Any]:
    """OPTIONAL: run a read-only query against PostgreSQL (requires ``psycopg``)."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise UnsupportedFormatError("PostgreSQL ingestion requires the optional 'psycopg' package") from exc
    with psycopg.connect(dsn) as con, con.cursor() as cur:  # pragma: no cover - needs a live server
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(query)
        names = [d.name for d in cur.description]
        rows = cur.fetchall()
    cols = list(zip(*rows)) if rows else [[] for _ in names]
    pq.write_table(pa.table({n: _to_arrow_array(list(c)) for n, c in zip(names, cols)}), dest)
    return {"reader": "postgres"}
