"""Rules found by testing on public schemas the system was never tuned on (experiments/run_unseen_schemas.py)."""
import random
import sqlite3

import duckdb
import pandas as pd

from backend.core.models import JoinKind
from backend.graph.relationships import _name_tokens, _tokens_agree
from backend.ingestion.csv_repair import repair
from backend.session.state import Session


def test_csv_with_unquoted_delimiters_in_text_is_repaired_not_collapsed(tmp_path):
    src = tmp_path / "shipments.csv"
    lines = ["id,customer,address,city,weight_kg,shipped"]
    rng = random.Random(3)
    for i in range(1, 201):
        street = rng.choice(["Main St 4", "Rua do Paço, 67", "2, rue du Commerce", "Obere Str. 57"])
        lines.append(f"{i},Cust {i},{street},{rng.choice(['Lyon', 'Berlin', 'Rio'])},{rng.uniform(1, 40):.2f},{'NULL' if i % 10 == 0 else '2024-05-0' + str(1 + i % 9)}")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    s = Session(workspace_root=tmp_path / "ws")
    art = s.add_file(src)
    assert art.column_count == 6, "a sniffer failure must not turn the file into one column"
    stats = art.metadata["csv_repair"]
    assert stats["merged"] > 0 and stats["unrepairable"] == 0
    rows = duckdb.sql(f"SELECT id, address, city, weight_kg, shipped FROM '{art.storage_path}' ORDER BY id").fetchall()
    assert len(rows) == 200
    assert {r[1] for r in rows} == {"Main St 4", "Rua do Paço, 67", "2, rue du Commerce", "Obere Str. 57"}
    assert all(r[2] in {"Lyon", "Berlin", "Rio"} for r in rows)
    assert all(isinstance(r[3], float) for r in rows)  # the numeric column stayed numeric
    assert sum(r[4] is None for r in rows) == 20  # "NULL" is a missing value


def test_repair_breaks_ties_with_the_comma_space_cue(tmp_path):
    src, dst = tmp_path / "a.csv", tmp_path / "b.csv"
    src.write_text("name,street,city\nAcme,Main St 1,Paris\nVins,2, rue X,Lyon\n", encoding="utf-8")
    repair(src, dst, ",")
    rows = list(duckdb.sql(f"SELECT * FROM read_csv('{dst}', header=true, all_varchar=true)").fetchall())
    assert rows[1] == ("Vins", "2, rue X", "Lyon")


def test_every_table_of_a_sqlite_database_is_loaded(tmp_path):
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE artist (ArtistId INTEGER PRIMARY KEY, Name TEXT)")
        con.execute("CREATE TABLE album (AlbumId INTEGER PRIMARY KEY, Title TEXT, ArtistId INTEGER)")
        con.executemany("INSERT INTO artist VALUES (?, ?)", [(i, f"artist {i}") for i in range(1, 51)])
        con.executemany("INSERT INTO album VALUES (?, ?, ?)", [(i, f"album {i}", 1 + i % 50) for i in range(1, 201)])
    s = Session(workspace_root=tmp_path / "ws")
    arts = s.add_files(db)
    assert {a.dataset_id for a in arts} == {"artist", "album"}
    assert {a.metadata["table"] for a in arts} == {"artist", "album"}
    s.discover()
    rel = next(r for r in s.relationships if {r.left_dataset, r.right_dataset} == {"artist", "album"})
    assert rel.join_kind == JoinKind.LOOKUP and rel.evidence["dimension"] == "artist"


def _surrogate_schema(tmp_path):
    """Four tables keyed by 1..n integers. Only products.category_id and orders.product_id are real references;
    the other key ranges overlap by construction."""
    rng = random.Random(9)
    frames = {
        "categories": pd.DataFrame({"category_id": range(1, 9), "category_name": [f"cat {i}" for i in range(1, 9)]}),
        "employees": pd.DataFrame({"employee_id": range(1, 10), "last_name": [f"Surname{i}" for i in range(1, 10)], "region": ["WA"] * 5 + ["OR"] * 4}),
        "products": pd.DataFrame({"product_id": range(1, 78), "category_id": [rng.randint(1, 8) for _ in range(77)], "unit_price": [rng.uniform(2, 90) for _ in range(77)]}),
        "orders": pd.DataFrame({"order_id": range(1, 801), "product_id": [rng.randint(1, 77) for _ in range(800)], "quantity": [rng.randint(1, 30) for _ in range(800)],
                                "region": [rng.choice(["WA", "OR", "CA"]) for _ in range(800)]}),
    }
    paths = []
    for name, df in frames.items():
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    s = Session(workspace_root=tmp_path / "ws")
    for p in paths:
        s.add_file(p)
    s.discover()
    return s


def test_overlapping_surrogate_key_ranges_are_not_relationships(tmp_path):
    s = _surrogate_schema(tmp_path)
    pairs = {frozenset((r.left_dataset, r.right_dataset)): r for r in s.relationships}
    assert frozenset(("categories", "employees")) not in pairs, "category_id 1..8 and employee_id 1..9 overlap only by range"
    assert pairs[frozenset(("products", "categories"))].join_kind == JoinKind.LOOKUP
    assert pairs[frozenset(("orders", "products"))].join_kind == JoinKind.LOOKUP


def test_shared_attribute_of_two_entity_tables_is_not_an_aggregate_join(tmp_path):
    s = _surrogate_schema(tmp_path)
    kinds = {(frozenset((r.left_dataset, r.right_dataset)), r.join_kind) for r in s.relationships}
    assert (frozenset(("orders", "employees")), JoinKind.AGGREGATE_LOOKUP) not in kinds, "employees and orders merely share a region value"


def test_name_tokens_point_references_to_their_tables():
    assert _tokens_agree(_name_tokens("shipVia"), _name_tokens("shippers"))
    assert _tokens_agree(_name_tokens("PULocationID"), _name_tokens("LocationID"))
    assert _tokens_agree(_name_tokens("categoryID"), _name_tokens("categories"))
    assert not _tokens_agree(_name_tokens("categoryID"), _name_tokens("employeeID"))
    assert not _tokens_agree(_name_tokens("shipperID"), _name_tokens("supplierID"))
