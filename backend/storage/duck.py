"""DuckDB helpers. Every query runs against Parquet files on disk so memory
stays bounded by DuckDB's buffer manager rather than dataset size."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import duckdb


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parquet_scan(path: str | Path) -> str:
    return f"read_parquet({quote_literal(Path(path).as_posix())})"


@contextmanager
def connect(memory_limit: str | None = None, threads: int | None = None):
    con = duckdb.connect(database=":memory:")
    try:
        if memory_limit is None:
            from backend.config import load_config

            memory_limit = (load_config().get("engine") or {}).get("memory_limit")
        if memory_limit:
            con.execute(f"SET memory_limit={quote_literal(memory_limit)}")
        if threads:
            con.execute(f"SET threads={int(threads)}")
        con.execute("SET preserve_insertion_order=false")
        yield con
    finally:
        con.close()
