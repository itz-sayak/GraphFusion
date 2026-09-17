"""Performance benchmark: 100K / 500K / 1M trip rows through the full pipeline.

Trips are uniform samples of the real TLC month (data/raw/nyc/...), joined with
the real zone, NTA and census sources — i.e. the actual NYC integration at
three scales. Each size runs in a fresh subprocess so peak memory (process
RSS, including DuckDB/Arrow native memory) is measured in isolation.

Stages timed: ingestion, profiling, schema matching + graph construction,
merge planning, merge execution (joins, aggregation, validation, provenance, Parquet export).

    python experiments/benchmark.py [--sizes 100000 500000 1000000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import RESULTS, ROOT, PeakMemory, line_chart, markdown_table, save_json


def run_one(size: int, memory_limit: str = "2GB") -> dict:
    import duckdb

    from backend.session.state import Session

    raw = ROOT / "data/raw/nyc/yellow_tripdata_2024-01.parquet"
    nyc = ROOT / "data/processed/nyc"
    tmp = Path(tempfile.mkdtemp(prefix=f"dfg_bench_{size}_"))
    try:
        trips = tmp / "yellow_tripdata_bench.parquet"
        duckdb.sql(f"COPY (SELECT * FROM read_parquet('{raw.as_posix()}') USING SAMPLE reservoir({size} ROWS) REPEATABLE (7)) TO '{trips.as_posix()}' (FORMAT parquet)")
        t: dict[str, float] = {}
        with PeakMemory() as mem:
            s = Session(workspace_root=tmp / "ws")
            s.config["engine"]["memory_limit"] = memory_limit
            start = time.perf_counter()
            arts = [s.add_file(f, profile=False) for f in (trips, nyc / "taxi_zone_lookup.csv", nyc / "nta_demographics.json", nyc / "census_acs_nyc_counties.json")]
            t["ingestion"] = time.perf_counter() - start
            start = time.perf_counter()
            for a in arts:
                s.profile(a.dataset_id)
            t["profiling"] = time.perf_counter() - start
            start = time.perf_counter()
            s.discover()
            t["matching_and_graph"] = time.perf_counter() - start
            start = time.perf_counter()
            s.make_plan()
            t["planning"] = time.perf_counter() - start
            start = time.perf_counter()
            rec = s.execute(write_csv=False)
            t["merge_validate_export"] = time.perf_counter() - start
        return {
            "rows": size, "memory_limit": memory_limit, "output_rows": rec.result.row_count, "validation_passed": rec.result.validation["passed"],
            **{k: round(v, 2) for k, v in t.items()}, "total": round(sum(t.values()), 2), "peak_rss_mb": mem.peak_mb,
            "rows_per_second": round(size / sum(t.values())), "merge_stage_ms": rec.result.stage_timings_ms,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--limit", default="2GB", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(args.one, args.limit)))
        return
    if not (ROOT / "data/raw/nyc/yellow_tripdata_2024-01.parquet").exists():
        raise SystemExit("Run python scripts/download_demo_data.py first")
    results = []
    runs = [(size, "2GB") for size in args.sizes] + [(max(args.sizes), "512MB")]
    for size, limit in runs:
        proc = subprocess.run([sys.executable, __file__, "--one", str(size), "--limit", limit], capture_output=True, text=True, encoding="utf-8")
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(f"benchmark for {size} failed:\n{proc.stderr[-3000:]}")
        r = json.loads(line[7:])
        results.append(r)
        print(f"  {size:>9,} rows @ {limit}: total {r['total']:.1f}s, peak RSS {r['peak_rss_mb']:.0f} MB, validation {r['validation_passed']}")
    import platform

    import psutil

    machine = {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform(), "python": platform.python_version()}
    save_json("benchmark.json", {"machine": machine, "results": results})
    cols = ["rows", "memory_limit", "ingestion", "profiling", "matching_and_graph", "planning", "merge_validate_export", "total", "peak_rss_mb", "rows_per_second"]
    md = ["# Performance benchmark (NYC integration at scale)", "", f"Machine: {machine}", "", markdown_table(results, cols, ["trip rows", "DuckDB memory limit", "ingest (s)", "profile (s)", "match+graph (s)", "plan (s)", "merge+validate+export (s)", "total (s)", "peak RSS (MB)", "rows/s"],
          {"rows": ",", "rows_per_second": ","}), "",
          "Profiling and schema matching hold only bounded samples and sketches, so their memory is flat in the number of rows. "
          "Merge execution memory is governed by `engine.memory_limit`: intermediates are lazy DuckDB views and the result is streamed to Parquet, "
          "so DuckDB uses memory up to the limit and spills to disk beyond it. The last row repeats the largest size with a 512MB limit: "
          "lower peak memory in exchange for a slower merge stage. Peak RSS includes the Python process baseline (~250-300MB)."]
    (RESULTS / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    results_2gb = [r for r in results if r["memory_limit"] == "2GB"]
    xs = [r["rows"] for r in results_2gb]
    line_chart(RESULTS / "benchmark_time.png", xs, {k.replace("_", " "): [r[k] for r in results_2gb] for k in ("profiling", "matching_and_graph", "merge_validate_export")}, "Pipeline stage time by input size", "trip rows", "seconds")
    line_chart(RESULTS / "benchmark_memory.png", xs, {"peak RSS (2GB DuckDB limit)": [r["peak_rss_mb"] for r in results_2gb]}, "Peak process memory by input size", "trip rows", "MB", value_fmt="{:.0f}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
