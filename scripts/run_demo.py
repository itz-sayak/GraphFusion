"""Headless end-to-end demonstration through the conversational agent.

    python scripts/run_demo.py --sample          # intentionally difficult customer datasets
    python scripts/run_demo.py --nyc             # real NYC mobility sources (run download_demo_data.py first)
    python scripts/run_demo.py --nyc --provider nvidia_nim

Every step goes through the same path as the UI: natural-language message →
agent → validated tool calls → session orchestrator. The script then asserts
that every required output artifact exists and is well-formed, so it doubles
as an end-to-end check:

    ingestion → schema discovery → relationship detection → graph construction →
    merge planning → merge execution → conflict reporting → provenance → export
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from backend.llm.agent import ChatAgent  # noqa: E402
from backend.llm.factory import make_provider  # noqa: E402
from backend.session.state import Session  # noqa: E402

SAMPLE = [ROOT / "data/sample/sample_customers.csv", ROOT / "data/sample/sample_customer_master.json", ROOT / "data/sample/sample_sales.parquet"]
NYC = [ROOT / "data/processed/nyc/yellow_tripdata_sample.parquet", ROOT / "data/processed/nyc/taxi_zone_lookup.csv", ROOT / "data/processed/nyc/nta_demographics.json", ROOT / "data/processed/nyc/census_acs_nyc_counties.json"]

SAMPLE_SCRIPT = [
    "Inspect the schemas of sample_customers",
    "Find relationships between the datasets",
    "Which columns refer to the same entity?",
    "Why do city and location match?",
    "Show me the merge plan",
    "Merge them.",
    "Show me conflicts",
    "Show me duplicate entities",
    "What percentage of rows were matched?",
    "Why did you choose this join?",
    "Where does city come from?",
    "Export the final dataset",
]
NYC_SCRIPT = [
    "Find relationships between the datasets",
    "What mappings did you infer?",
    "How is yellow_tripdata_sample connected to census_acs_nyc_counties? Show the route",
    "Show me the merge plan",
    "Merge everything.",
    "What percentage of rows were matched?",
    "Generate a data lineage report",
    "Export the final dataset",
]
REQUIRED = ["unified_dataset.parquet", "unified_dataset.csv", "merge_report.json", "provenance.json", "conflicts.csv", "schema_mapping.json", "integration_graph.json"]


def banner(text: str) -> None:
    print("\n" + "═" * 100 + f"\n{text}\n" + "═" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--nyc", action="store_true")
    ap.add_argument("--provider", default="mock", help="mock | nvidia_nim | xai_grok | openai_compat | anthropic | ollama")
    ap.add_argument("--workspace", default=str(ROOT / "workspace"))
    ap.add_argument("--quiet", action="store_true", help="print only tool calls, not full replies")
    args = ap.parse_args()

    files, script = (SAMPLE, SAMPLE_SCRIPT) if args.sample else (NYC, NYC_SCRIPT)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        hint = "python scripts/make_sample_data.py" if args.sample else "python scripts/download_demo_data.py"
        print(f"Missing input files: {missing}\nRun: {hint}")
        return 2

    started = time.perf_counter()
    session = Session(workspace_root=args.workspace)
    agent = ChatAgent(make_provider(args.provider))
    banner(f"GraphFusion demo — session {session.session_id} — provider {agent.provider.describe()}")
    print("User: Load these datasets.")
    for f in files:
        art = session.add_file(f)
        print(f"  loaded {art.source_name}: {art.row_count:,} rows × {art.column_count} columns ({art.source_type.value})")

    for message in script:
        banner(f"User: {message}")
        res = agent.handle(session, message)
        calls = ", ".join(f"{t['tool']}({json.dumps(t['arguments'])})" + ("" if t["ok"] else f" ✖ {t['error']}") for t in res["tool_calls"])
        print(f"[tools] {calls or '—'}   [{res['elapsed_ms'] / 1000:.1f}s]")
        if not args.quiet:
            reply = res["reply"]
            print(reply if len(reply) < 6000 else reply[:6000] + "\n…(truncated)")

    banner("End-to-end verification")
    rec = session.current_merge()
    ok = True
    for name in REQUIRED:
        path = Path(rec.files.get(name, ""))
        exists = path.is_file() and path.stat().st_size > 0
        ok &= exists
        print(f"  {'✔' if exists else '✖'} {name:28s} {path.stat().st_size if exists else 0:>12,} bytes")
    for name in ("merge_report.json", "provenance.json", "schema_mapping.json", "integration_graph.json"):
        json.loads(Path(rec.files[name]).read_text(encoding="utf-8"))
    rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{Path(rec.files['unified_dataset.parquet']).as_posix()}')").fetchone()[0]
    print(f"  unified rows: {rows:,}; validation passed: {rec.result.validation['passed']} ({len(rec.result.validation['checks'])} checks)")
    ok &= rec.result.validation["passed"] and rows == rec.result.row_count
    print(f"  output directory: {Path(rec.files['unified_dataset.parquet']).parent}")
    print(f"\n{'PASS' if ok else 'FAIL'} — total {time.perf_counter() - started:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
