import time
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from backend.core.errors import InvalidStateError
from backend.provenance.uncertainty import aggregate_with_uncertainty
from backend.session.state import Session


def test_scenarios_drop_rows_joined_through_an_uncertain_relationship(tmp_path):
    path = (tmp_path / "u.parquet").as_posix()
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES
            ('north', 10.0, 1.0, 7, true),
            ('north', 20.0, 1.0, NULL, false),
            ('south', 5.0, 1.0, 3, true)) t(region, amount, _match_probability, _src_dim_row, _nested_match))
        TO '{path}' (FORMAT parquet)""")
    scenarios = [{"relationship": "facts ← dim", "flag_column": None, "src_row_column": "_src_dim_row", "confidence": 0.8},
                 {"relationship": "dim ← geo", "flag_column": "_nested_match", "src_row_column": None, "confidence": 0.6}]
    res = aggregate_with_uncertainty(path, "amount", "region", "sum", 0.5, 10, scenarios)
    by = {s["relationship"]: s for s in res["scenarios"]}
    assert by["facts ← dim"]["probability_wrong"] == pytest.approx(0.2)
    assert by["facts ← dim"]["if_wrong"] == {"north": 20.0, "south": 0.0}  # only the unmatched row stays
    assert by["dim ← geo"]["if_wrong"]["north"] == 20.0


def test_history_as_known_at_returns_what_an_earlier_merge_recorded(tmp_path, sample_files):
    s = Session(workspace_root=tmp_path)
    for f in sample_files:
        s.add_file(f)
    first = s.execute(write_csv=False)
    between = datetime.now(timezone.utc)
    time.sleep(0.01)
    s.set_preferences(conflict_strategy="prefer_latest")
    second = s.execute(write_csv=False)
    assert s.entity_history(limit=5)["merge_id"] == second.result.merge_id
    old = s.entity_history(limit=5, as_known_at=between.isoformat())
    assert old["merge_id"] == first.result.merge_id and old["rows"]
    assert all("recorded_at" in r for r in old["rows"])
    with pytest.raises(InvalidStateError):
        s.entity_history(as_known_at=(first.result.created_at - timedelta(days=1)).isoformat())


def test_adaptive_blocking_splits_oversized_name_blocks_but_keeps_variants():
    from backend.core.models import SemanticType as S
    from backend.entity_resolution.blocking import generate_candidates

    cities = [f"city{i}" for i in range(20)]
    records = {(0, i): {"name": f"Rahul{i % 7} Sharma", "city": cities[i % 20]} for i in range(600)}
    records[(1, 0)] = {"name": "R. Sharma", "city": "city3"}  # abbreviation of a (0, i) record in city3
    fields = [("name", S.NAME, "basic"), ("city", S.CITY, "basic")]
    kw = dict(allow_within={0}, max_block_size=5000, window=3, schemes=["token"])
    full, _ = generate_candidates(records, fields, **kw)
    refined, stats = generate_candidates(records, fields, refine_block_size=100, **kw)
    assert stats.refined_blocks >= 1
    assert len(refined) < len(full) / 5
    assert ((0, 3), (1, 0)) in refined, "same surname/initial in the same city is still compared"
