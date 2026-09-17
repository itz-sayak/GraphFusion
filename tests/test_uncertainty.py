import math

import duckdb
import pytest

from backend.core.errors import ToolArgumentError
from backend.provenance.uncertainty import aggregate_with_uncertainty


@pytest.fixture()
def table(tmp_path):
    path = (tmp_path / "u.parquet").as_posix()
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES ('a', 10.0, 1.0, 0.8), ('a', 20.0, 0.5, 0.8), ('b', 5.0, 0.95, 1.0), ('b', NULL, 0.2, 1.0))
                       t(grp, amount, _match_probability, _relationship_confidence)) TO '{path}' (FORMAT parquet)""")
    return path


def test_expected_value_variance_and_bounds(table):
    res = aggregate_with_uncertainty(table, "amount", "grp", "sum", certain_threshold=0.9)
    a = next(g for g in res["groups"] if g["group"] == "a")
    assert a["point"] == 30.0
    assert a["expected"] == pytest.approx(1.0 * 10 + 0.5 * 20)
    sd = math.sqrt(0.5 * 0.5 * 400)
    assert a["ci95_low"] == pytest.approx(20 - 1.96 * sd, abs=1e-3) and a["ci95_high"] == pytest.approx(20 + 1.96 * sd, abs=1e-3)
    assert a["certain_only"] == 10.0 <= a["expected"] <= a["point"]
    assert a["min_relationship_confidence"] == 0.8
    assert res["rows"] == 4 and res["expected_correct_rows"] == pytest.approx(2.65) and res["rows_below_threshold"] == 2


def test_count_and_avg(table):
    cnt = {g["group"]: g for g in aggregate_with_uncertainty(table, None, "grp", "count")["groups"]}
    assert cnt["b"]["point"] == 2 and cnt["b"]["expected"] == pytest.approx(1.15)
    avg = {g["group"]: g for g in aggregate_with_uncertainty(table, "amount", "grp", "avg")["groups"]}
    assert avg["a"]["point"] == 15.0 and avg["a"]["expected"] == pytest.approx(20 / 1.5, abs=1e-3) and avg["a"]["ci95_low"] is None


def test_argument_validation(table):
    with pytest.raises(ToolArgumentError):
        aggregate_with_uncertainty(table, "nope", None, "sum")
    with pytest.raises(ToolArgumentError):
        aggregate_with_uncertainty(table, None, None, "sum")


def test_merge_output_carries_probabilities(merged_session):
    s = merged_session
    rec = s.current_merge()
    path = rec.files["unified_dataset.parquet"]
    lo, hi = duckdb.execute("SELECT min(_match_probability), max(_match_probability) FROM read_parquet(?)", [path]).fetchone()
    assert 0 < lo <= hi <= 1
    assert rec.result.quality_report["expected_correct_rows"] <= rec.result.row_count
    res = s.aggregate_with_uncertainty(None, None, "count")
    assert res["groups"][0]["point"] == rec.result.row_count
    assert res["groups"][0]["expected"] == pytest.approx(rec.result.quality_report["expected_correct_rows"], abs=0.01)
