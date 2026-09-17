import json
from pathlib import Path

import duckdb


def test_column_lineage_traces_to_sources(merged_session):
    lin = merged_session.provenance()["columns"]
    city = lin["city"]
    assert {(s["dataset"], s["column"]) for s in city["sources"]} == {("sample_customers", "city"), ("sample_customer_master", "location")}
    assert [h["operation"] for h in city["path"]][:2] == ["lookup", "fuse"]
    usd = lin["amount_spent_usd"]
    assert usd["sources"][0]["column"] == "amount_spent_usd" and "convert_to_USD" in usd["sources"][0]["transformation"]


def test_row_and_cell_lineage(merged_session):
    rec = merged_session.current_merge()
    out = rec.result.output_path.replace("\\", "/")
    cols = {r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()}
    assert {"_record_id", "_src_sample_sales_row", "_src_sample_customers_row", "_match_sample_customers"} <= cols
    cells = duckdb.sql(f"SELECT * FROM read_parquet('{Path(rec.files['cell_provenance.parquet']).as_posix()}') LIMIT 5").fetchall()
    assert cells, "fused cells must carry source-record provenance"


def test_prov_document(merged_session):
    rec = merged_session.current_merge()
    prov = json.loads(Path(rec.files["provenance.json"]).read_text(encoding="utf-8"))["prov"]
    assert prov["prefix"]["prov"] == "http://www.w3.org/ns/prov#"
    assert any(k.startswith("dfg:source/") and v["dfg:sha256"] for k, v in prov["entity"].items())
    assert any(v["prov:usedEntity"] == "dfg:source/sample_sales" for v in prov["wasDerivedFrom"].values())


def test_lineage_report_text(merged_session):
    report = merged_session.lineage_report()
    assert "unified.city" in report and "sample_customer_master.location" in report
