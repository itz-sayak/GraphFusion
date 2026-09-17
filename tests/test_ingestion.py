import json
import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.core.errors import IngestionError, UnsupportedFormatError
from backend.core.models import SourceType
from backend.ingestion.detect import detect_format
from backend.ingestion.loader import ingest_file, make_dataset_id
from backend.ingestion.readers import extract_records
from backend.storage.workspace import sha256_file


def test_detect_formats(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_path / "b.json").write_text('[{"x": 1}, {"x": 2}]')
    (tmp_path / "c.jsonl").write_text('{"x": 1}\n{"x": 2}\n')
    (tmp_path / "mislabelled.json").write_text('{"x": 1}\n{"x": 2}\n{"x": 3}\n')
    pq.write_table(pa.table({"x": [1, 2]}), tmp_path / "d.bin")
    pd.DataFrame({"x": [1]}).to_excel(tmp_path / "e.xlsx", index=False)
    assert detect_format(tmp_path / "a.csv") == SourceType.CSV
    assert detect_format(tmp_path / "b.json") == SourceType.JSON
    assert detect_format(tmp_path / "c.jsonl") == SourceType.JSONL
    assert detect_format(tmp_path / "mislabelled.json") == SourceType.JSONL  # content beats extension
    assert detect_format(tmp_path / "d.bin") == SourceType.PARQUET  # magic bytes beat extension
    assert detect_format(tmp_path / "e.xlsx") == SourceType.XLSX


def test_unsupported_format(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(bytes(range(256)))
    with pytest.raises(UnsupportedFormatError):
        detect_format(p)


def test_csv_is_lossless_for_mixed_dates_and_leading_zeros(workspace, tmp_path):
    src = tmp_path / "people.csv"
    src.write_text("zip,signup,amount\n00501,2024-01-05,10\n10001,05/02/2024,11.5\n02134,23/03/2024,\n")
    art = ingest_file(workspace, src)
    df = pq.read_table(art.storage_path).to_pandas()
    assert list(df["zip"]) == ["00501", "10001", "02134"], "leading zeros must survive"
    assert list(df["signup"]) == ["2024-01-05", "05/02/2024", "23/03/2024"], "mixed date formats kept as text, not coerced"
    assert art.metadata["promoted_types"] == {"amount": "DOUBLE"}
    assert art.row_count == 3 and art.column_count == 3


def test_json_envelopes(workspace, tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps([["NAME", "B01003_001E", "county"], ["Kings County, New York", "2631580", "047"]]))
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"meta": {"v": 1}, "records": [{"id": 1, "address": {"city": "Pune", "zip": "411001"}, "tags": ["a"]}]}))
    a = ingest_file(workspace, census)
    b = ingest_file(workspace, nested)
    assert a.row_count == 1 and set(a.schema) == {"NAME", "B01003_001E", "county"}
    assert {"id", "address.city", "address.zip", "tags"} <= set(b.schema)


def test_extract_records_rejects_scalars():
    with pytest.raises(IngestionError):
        extract_records([1, 2, 3])


def test_jsonl_xlsx_sqlite(workspace, tmp_path):
    (tmp_path / "e.jsonl").write_text('{"id": 1, "name": "A"}\n{"id": 2, "name": "B"}\n')
    pd.DataFrame({"id": [1, 2, 3], "city": ["Pune", "Delhi", None]}).to_excel(tmp_path / "t.xlsx", index=False)
    db = tmp_path / "t.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
        con.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "x"), (2, "y")])
    assert ingest_file(workspace, tmp_path / "e.jsonl").row_count == 2
    assert ingest_file(workspace, tmp_path / "t.xlsx").row_count == 3
    art = ingest_file(workspace, db)
    assert art.source_type == SourceType.SQLITE and art.row_count == 2


def test_originals_untouched_and_checksummed(workspace, sample_files):
    before = {f: sha256_file(f) for f in sample_files}
    arts = [ingest_file(workspace, f) for f in sample_files]
    for f, a in zip(sample_files, arts):
        assert sha256_file(f) == before[f] == a.checksum
    assert {a.source_type for a in arts} == {SourceType.CSV, SourceType.JSON, SourceType.PARQUET}


def test_dataset_ids_are_unique():
    assert make_dataset_id("Customer Master.parquet") == "customer_master"
    assert make_dataset_id("customers.csv", {"customers"}) == "customers_2"
