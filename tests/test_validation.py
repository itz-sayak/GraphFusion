import os
import stat
from pathlib import Path

from backend.validation.validators import validate_merge


def test_all_integrity_checks_pass(merged_session):
    v = merged_session.current_merge().result.validation
    names = {c["check"] for c in v["checks"]}
    assert v["passed"]
    for required in ("row_conservation", "no_fan_out", "root_values_preserved"):
        assert required in names
    assert any(n.startswith("source_immutable[") for n in names)
    assert any(n.startswith("entity_assignment[") for n in names)


def test_quality_report_fields(merged_session):
    q = merged_session.current_merge().result.quality_report
    for key in ("rows_ingested", "schema_mappings_by_band", "resolved_entities", "conflicts", "duplicate_entities_resolved", "integration_coverage"):
        assert key in q
    assert "DATA QUALITY REPORT" in q["text"]
    assert 0 < q["integration_coverage"] <= 1


def test_tampered_source_is_detected(tmp_path, sample_files):
    from backend.session.state import Session
    import duckdb
    from backend.merge.executor import MergeExecutor

    s = Session(workspace_root=tmp_path)
    for f in sample_files:
        s.add_file(f)
    s.discover()
    plan = s.make_plan()
    con = duckdb.connect()
    ctx = MergeExecutor(plan, s.artifacts, s.profiles, s.config, con).run()
    raw = Path(s.artifacts["sample_sales"].metadata["raw_copy"])
    os.chmod(raw, stat.S_IWRITE | stat.S_IREAD)
    with open(raw, "ab") as fh:
        fh.write(b"tamper")
    result = validate_merge(con, ctx, plan, s.artifacts)
    assert not result["passed"]
    assert any(c["check"] == "source_immutable[sample_sales]" and not c["passed"] for c in result["checks"])
