import duckdb
import pytest

from backend.core.errors import ToolArgumentError
from backend.merge.canonical import NameAllocator, role_for
from backend.merge.modes import effective_thresholds
from backend.session.state import Session


def test_roles_and_names():
    assert role_for("PULocationID", "LocationID") == "pickup"
    assert role_for("DOLocationID", "LocationID") == "dropoff"
    assert role_for("cust_id", "customer_id") is None
    alloc = NameAllocator()
    assert alloc.allocate("city", "a") == "city"
    assert alloc.allocate("city", "b") == "b_city"


def test_modes_and_overrides(config):
    strict = effective_thresholds(config, "strict")
    permissive = effective_thresholds(config, "permissive")
    assert strict["auto_merge"] > permissive["auto_merge"]
    t = effective_thresholds(config, "balanced", {"confidence_threshold": 0.95, "merge_uncertain": False})
    assert t["auto_merge"] == 0.95 and t["min_edge"] == 0.95
    with pytest.raises(ToolArgumentError):
        effective_thresholds(config, "yolo")


def test_plan_is_explicit(merged_session):
    plan = merged_session.make_plan()
    assert plan.root_dataset == "sample_sales"
    ops = [s.operation for s in plan.steps]
    assert "resolve_entities" in ops and "lookup" in ops and "transform" in ops
    lookup = next(s for s in plan.steps if s.operation == "lookup")
    assert lookup.join_type == "left" and lookup.join_keys and lookup.threshold is not None
    names = {c.name for c in plan.canonical_schema}
    assert {"customer_name", "city", "email", "entity_id", "amount_spent_usd"} <= names


def test_merge_preserves_grain_and_never_drops_rows(merged_session):
    rec = merged_session.current_merge()
    out = rec.result.output_path.replace("\\", "/")
    sales_rows = merged_session.artifacts["sample_sales"].row_count
    total, distinct, unmatched = duckdb.sql(
        f"SELECT count(*), count(DISTINCT _src_sample_sales_row), count(*) FILTER (WHERE NOT _match_sample_customers) FROM read_parquet('{out}')"
    ).fetchone()
    assert total == distinct == sales_rows
    assert unmatched > 0, "orphan sales rows are kept, flagged, not dropped"
    assert rec.result.validation["passed"]


def test_transformations_applied(merged_session):
    out = merged_session.current_merge().result.output_path.replace("\\", "/")
    row = duckdb.sql(f"SELECT amount_spent_amount, amount_spent_currency, amount_spent_usd, amount_spent_raw, txn_date FROM read_parquet('{out}') WHERE amount_spent_raw LIKE '₹%' LIMIT 1").fetchone()
    assert row[1] == "INR" and row[2] < row[0] and row[3].startswith("₹") and row[4] is not None
    bad_dates = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out}') WHERE txn_date IS NULL AND txn_date_raw IS NOT NULL").fetchone()[0]
    assert bad_dates == 0


def test_entity_grain_when_primary_key_pinned(tmp_path, sample_files):
    s = Session(workspace_root=tmp_path)
    for f in sample_files[:2]:
        s.add_file(f)
    rec = s.execute()
    assert rec.plan.grain.startswith("one row per resolved entity")
    er = rec.group_stats["G1"]
    assert rec.result.row_count == er["entities"] < er["records"]


def test_strict_threshold_leaves_weak_relationship_unmerged(tmp_path, sample_files):
    s = Session(workspace_root=tmp_path)
    for f in sample_files:
        s.add_file(f)
    s.set_preferences(confidence_threshold=95)
    plan = s.make_plan()
    assert "sample_customer_master" in plan.unmerged_datasets
    assert any("left unmerged (not discarded)" in w for w in plan.warnings)
