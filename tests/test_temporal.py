from datetime import datetime

import duckdb

from backend.merge.temporal import DatedValue, build_intervals, check_intervals


def _v(value, day, kind="change", ds="crm", priority=1):
    return DatedValue(value, value.lower() if value else None, ds, "city", 0, datetime(2024, 1, day) if day else None, kind if day else None, priority)


def test_intervals_merge_runs_and_close_on_change():
    hist = build_intervals("E1", "city", [_v("Pune", 1), _v("pune", 5), _v("Bangalore", 9), _v("Delhi", None)])
    dated = [h for h in hist if h["valid_from"]]
    assert [h["value"] for h in dated] == ["Pune", "Bangalore"]
    assert dated[0]["valid_to"] == datetime(2024, 1, 9) and dated[0]["supporting_records"] == 2
    assert dated[1]["is_current"] and dated[1]["valid_to"] is None
    assert [h["timestamp_kind"] for h in hist if not h["valid_from"]] == ["undated"]
    assert check_intervals(hist) == []


def test_same_timestamp_tie_goes_to_priority_and_stays_consistent():
    hist = build_intervals("E1", "city", [_v("Pune", 3, ds="crm", priority=2), _v("Mumbai", 3, ds="mdm", priority=0)])
    assert [h["value"] for h in hist] == ["Mumbai"]
    assert check_intervals(hist) == []


def test_check_intervals_detects_overlap():
    bad = [
        {"entity_id": "E", "attribute": "a", "valid_from": datetime(2024, 1, 1), "valid_to": datetime(2024, 1, 10), "is_current": False},
        {"entity_id": "E", "attribute": "a", "valid_from": datetime(2024, 1, 5), "valid_to": None, "is_current": True},
    ]
    assert check_intervals(bad)


def test_history_export_as_of_and_prefer_latest(merged_session):
    s = merged_session
    rec = s.current_merge()
    assert any(k.startswith("entity_history_") for k in rec.files)
    assert any(c["check"].startswith("temporal_consistency") and c["passed"] for c in rec.result.validation["checks"])
    h = s.entity_history(limit=5000)
    assert h["attributes_with_changes"] > 0
    past = next(r for r in h["rows"] if r["valid_to"])
    as_of = s.entity_history(past["entity_id"], past["attribute"], as_of=past["valid_from"])["rows"]
    assert [r["value"] for r in as_of] == [past["value"]], "as-of query returns the value valid at that time"

    # prefer_latest must agree with the history's current value whenever that value comes from a change timestamp
    s.set_preferences(conflict_strategy="prefer_latest")
    rec = s.execute()
    current = {(r["entity_id"], r["attribute"]): r["value"] for r in s.entity_history(limit=5000)["rows"] if r["is_current"] and r["timestamp_kind"] == "change"}
    assert current
    path = rec.files[next(k for k in rec.files if k.startswith("entities_"))]
    con = duckdb.connect()
    checked = 0
    for (ent, attr), value in current.items():
        got = con.execute(f'SELECT CAST("{attr}" AS VARCHAR) FROM read_parquet(?) WHERE entity_id = ?', [path, ent]).fetchone()
        if got is not None:
            assert got[0] == value, (ent, attr, got[0], value)
            checked += 1
    assert checked
