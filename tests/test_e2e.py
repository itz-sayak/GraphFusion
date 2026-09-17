"""End-to-end: a full conversation through the agent, from loading to export."""
import json
from pathlib import Path

import duckdb
import pytest

from backend.llm.agent import ChatAgent
from backend.llm.providers.mock import MockProvider
from backend.session.state import Session

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
def test_conversational_integration(tmp_path, sample_files):
    s = Session(workspace_root=tmp_path)
    agent = ChatAgent(MockProvider())
    for f in sample_files:
        s.add_file(f)

    steps = [
        ("Find relationships between the datasets", "find_candidate_relationships"),
        ("Show me the merge plan", "generate_merge_plan"),
        ("Merge them.", "execute_merge"),
        ("Show me conflicts", "show_conflicts"),
        ("Where does city come from?", "show_provenance"),
        ("Export the final dataset", "export_dataset"),
    ]
    for message, tool in steps:
        res = agent.handle(s, message)
        assert res["tool_calls"] and res["tool_calls"][-1]["tool"] == tool and all(t["ok"] for t in res["tool_calls"]), res

    rec = s.current_merge()
    for name in ("unified_dataset.parquet", "unified_dataset.csv", "merge_report.json", "provenance.json", "conflicts.csv", "schema_mapping.json", "integration_graph.json"):
        assert Path(rec.files[name]).stat().st_size > 0
    report = json.loads(Path(rec.files["merge_report.json"]).read_text(encoding="utf-8"))
    assert report["result"]["validation"]["passed"]
    rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{Path(rec.files['unified_dataset.parquet']).as_posix()}')").fetchone()[0]
    assert rows == s.artifacts["sample_sales"].row_count

    res = agent.handle(s, "Undo the previous merge")
    assert res["tool_calls"][0]["tool"] == "undo_merge"
    assert all(r.status == "undone" for r in s.merges)


@pytest.mark.e2e
@pytest.mark.skipif(not (ROOT / "data/processed/nyc/yellow_tripdata_sample.parquet").exists(), reason="NYC demo data not downloaded")
def test_nyc_real_world_integration(tmp_path):
    s = Session(workspace_root=tmp_path)
    nyc = ROOT / "data/processed/nyc"
    for f in ("yellow_tripdata_sample.parquet", "taxi_zone_lookup.csv", "nta_demographics.json", "census_acs_nyc_counties.json"):
        s.add_file(nyc / f)
    s.discover()
    kinds = {(frozenset((r.left_dataset, r.right_dataset)), r.join_kind.value) for r in s.relationships}
    assert (frozenset(("taxi_zone_lookup", "yellow_tripdata_sample")), "lookup") in kinds
    assert not any({"census_acs_nyc_counties", "taxi_zone_lookup"} == set(k) for k, _ in kinds), "coincidental integer overlap must not create a relationship"
    route = s.route("yellow_tripdata_sample", "census_acs_nyc_counties")
    assert route["found"] and route["uses_intermediate"] and route["direct_confidence"] is None
    rec = s.execute(write_csv=False)
    assert rec.result.row_count == s.artifacts["yellow_tripdata_sample"].row_count
    assert rec.result.validation["passed"]
    cols = set(rec.result.columns)
    assert {"pickup_borough", "pickup_zone", "dropoff_borough", "dropoff_zone", "pickup_median_household_income_avg"} <= cols
    out = rec.result.output_path.replace("\\", "/")
    manhattan = duckdb.sql(f"SELECT DISTINCT pickup_total_population_2010_number_sum FROM read_parquet('{out}') WHERE pickup_borough = 'Manhattan'").fetchall()
    assert manhattan == [(1585873.0,)], "borough population must equal the 2010 Census count, not a per-row multiple"
