"""Download NYC TLC Yellow Taxi trip records (Parquet) and the Taxi Zone Lookup (CSV).

Source: NYC Taxi & Limousine Commission, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

* One month of yellow-taxi trips (~50 MB, ~3M rows) is downloaded once into
  ``data/raw/nyc/`` and a reproducible, uniformly random subset (default
  500 000 rows, seeded reservoir sample) is written to ``data/processed/nyc/``.
  The sampling happens inside DuckDB directly over the Parquet file, so the
  full month is never loaded into Python memory.
* The taxi zone lookup (265 rows) is copied as-is.

The TLC schema drifts over time; the month is pinned and the expected columns
are validated so a silent upstream change fails loudly.
"""
from __future__ import annotations

import argparse
import shutil

import duckdb

from _common import PROCESSED, RAW, download

TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
EXPECTED = {"tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "PULocationID", "DOLocationID", "fare_amount", "tip_amount", "total_amount", "payment_type"}


def main(month: str, rows: int, seed: int, force: bool) -> None:
    print(f"NYC TLC yellow taxi trips {month}")
    raw = download(TRIPS_URL.format(month=month), RAW / f"yellow_tripdata_{month}.parquet", force=force)
    zones = download(ZONES_URL, RAW / "taxi_zone_lookup.csv", force=force)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "yellow_tripdata_sample.parquet"
    with duckdb.connect() as con:
        cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{raw.as_posix()}')").fetchall()}
        missing = EXPECTED - cols
        if missing:
            raise SystemExit(f"TLC schema changed: missing columns {sorted(missing)}")
        total = con.execute(f"SELECT count(*) FROM read_parquet('{raw.as_posix()}')").fetchone()[0]
        n = min(rows, total)
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{raw.as_posix()}') USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})) "
            f"TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"
        )
        written = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    print(f"  sampled {written:,} of {total:,} trips → {out.name}")
    shutil.copyfile(zones, PROCESSED / "taxi_zone_lookup.csv")
    print(f"  zone lookup → {PROCESSED / 'taxi_zone_lookup.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", default="2024-01", help="YYYY-MM (default 2024-01)")
    ap.add_argument("--rows", type=int, default=500_000, help="subset size (default 500000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    a = ap.parse_args()
    main(a.month, a.rows, a.seed, a.force)
