"""Download two public relational databases for the unseen-schema evaluation (testing only).

* Northwind (8 CSV files, neo4j-contrib/northwind-neo4j): orders/sales, small integer surrogate keys,
  and rows with unquoted commas inside addresses.
* Chinook (one SQLite file with 11 tables, lerocha/chinook-database): music store with two fact tables.

Files land in data/raw/northwind/ and data/raw/chinook/ (git-ignored); files already present are skipped.

    python scripts/download_unseen_schemas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NORTHWIND = "https://raw.githubusercontent.com/neo4j-contrib/northwind-neo4j/master/data/{}.csv"
NORTHWIND_TABLES = ["categories", "customers", "employees", "order-details", "orders", "products", "shippers", "suppliers"]
CHINOOK = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"


def fetch(client: httpx.Client, url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {dest.relative_to(ROOT)} (already present)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    print(f"  ✔ {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for t in NORTHWIND_TABLES:
            fetch(client, NORTHWIND.format(t), RAW / "northwind" / f"{t}.csv")
        fetch(client, CHINOOK, RAW / "chinook" / "Chinook_Sqlite.sqlite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
