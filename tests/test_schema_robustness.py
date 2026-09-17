"""Robustness of relationship inference on unseen, deliberately confusing schemas.

Each scenario is generated with meaningless table/column names (so names cannot help), one of
several identifier formats, and the traps that broke earlier versions on real data:

* identifier columns with the same format but disjoint values (must not be linked),
* a date column in a child table that looks like a parent's date but holds different values,
* a child whose reference column is near-unique (a few repeated parents) — must be N:1, not 1:1,
* sibling tables that share a parent key — must connect through the parent, not to each other,
* integer keys with overlapping ranges (coincidental numeric overlap).

Ground truth (as unordered dataset pairs):  items→orders, items→products, pay→orders, rev→orders, orders→cust
"""
from __future__ import annotations

import random
import string
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backend.session.state import Session

FORMATS = ["uuid_hex", "uuid_dashed", "prefixed", "integer"]


def _ids(fmt: str, n: int, rng: random.Random, prefix: str, offset: int = 0) -> list:
    if fmt == "uuid_hex":
        return [uuid.UUID(int=rng.getrandbits(128)).hex for _ in range(n)]
    if fmt == "uuid_dashed":
        return [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(n)]
    if fmt == "prefixed":
        return [f"{prefix}-{i + offset:07d}" for i in range(n)]
    return list(range(1 + offset, n + 1 + offset))


def _name(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(3)) + "_" + "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(4))


def make_scenario(out: Path, seed: int, fmt: str) -> dict[str, str]:
    rng = random.Random(seed)
    n_cust, n_orders, n_prod = 900, 1500, 300
    t = {k: _name(rng) for k in ("cust", "orders", "prod", "items", "pay", "rev")}
    c = {k: _name(rng) for k in ("cust_key", "order_key", "prod_key", "rev_key", "item_seq", "city", "status", "odate", "rdate", "price", "amount", "stars", "weight")}
    cust = _ids(fmt, n_cust, rng, "CU")
    orders = _ids(fmt, n_orders, rng, "OR", offset=5000 if fmt == "integer" else 0)
    prods = _ids(fmt, n_prod, rng, "PR", offset=90000 if fmt == "integer" else 0)
    base = date(2021, 1, 1)
    odates = [base + timedelta(days=rng.randrange(700)) for _ in orders]
    frames = {
        "cust": pd.DataFrame({c["cust_key"]: cust, c["city"]: [rng.choice(["alpha", "beta", "gamma", "delta", "epsilon"]) for _ in cust]}),
        "orders": pd.DataFrame({c["order_key"]: orders, c["cust_key"] + "x": [rng.choice(cust) for _ in orders],
                                c["status"]: [rng.choice(["new", "paid", "sent"]) for _ in orders], c["odate"]: odates}),
        "prod": pd.DataFrame({c["prod_key"]: prods, c["weight"]: [round(rng.uniform(0.1, 30), 2) for _ in prods]}),
    }
    items = [(o, i + 1, rng.choice(prods), round(rng.uniform(5, 500), 2)) for o in orders for i in range(rng.choice([1, 1, 1, 2, 3]))]
    frames["items"] = pd.DataFrame(items, columns=[c["order_key"] + "y", c["item_seq"], c["prod_key"] + "y", c["price"]])
    frames["pay"] = pd.DataFrame([(o, round(rng.uniform(5, 900), 2)) for o in orders for _ in range(rng.choice([1, 1, 2]))], columns=[c["order_key"] + "z", c["amount"]])
    # reviews: one per order, plus a few repeated orders (near-unique reference); own id with the SAME format as the other keys
    rev_orders = orders + [rng.choice(orders) for _ in range(8)]
    rev_ids = _ids(fmt, len(rev_orders), rng, "RV", offset=700000 if fmt == "integer" else 0)
    # a date column that looks like the order date (same range, same format) but is unrelated per row
    rdates = [base + timedelta(days=rng.randrange(700)) for _ in rev_orders]
    frames["rev"] = pd.DataFrame({c["rev_key"]: rev_ids, c["order_key"] + "w": rev_orders, c["rdate"]: rdates, c["stars"]: [rng.randint(1, 5) for _ in rev_orders]})
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, df in frames.items():
        p = out / f"{t[k]}.csv"
        df.to_csv(p, index=False)
        paths[k] = p.stem
    return paths


@pytest.mark.parametrize("seed,fmt", [(seed, fmt) for seed, fmt in zip(range(1, 9), FORMATS * 2)])
def test_relationships_on_unseen_confusing_schemas(tmp_path, seed, fmt):
    names = make_scenario(tmp_path / "data", seed, fmt)
    s = Session(workspace_root=tmp_path / "ws")
    for f in sorted((tmp_path / "data").glob("*.csv")):
        s.add_file(f)
    s.discover()
    ds = {v: k for k, v in names.items()}
    found = {frozenset((ds[r.left_dataset], ds[r.right_dataset])): r for r in s.relationships}
    truth = {frozenset(p) for p in (("items", "orders"), ("items", "prod"), ("pay", "orders"), ("rev", "orders"), ("orders", "cust"))}
    missing = truth - set(found)
    extra = {tuple(sorted(p)) for p in set(found) - truth}
    assert not missing, f"{fmt}: missed {missing}"
    # siblings (items/pay/rev share the order key) must not be linked directly
    assert not extra & {("items", "pay"), ("items", "rev"), ("pay", "rev")}, f"{fmt}: sibling links {extra}"
    # the near-unique review reference is many-to-one, never a 1:1 entity merge
    assert found[frozenset(("rev", "orders"))].join_kind.value == "lookup", f"{fmt}: {found[frozenset(('rev', 'orders'))].join_kind}"
    # no accepted correspondence between different identifier systems, nor between the unrelated date columns
    for m in s.matching.matches:
        if not m.accepted:
            continue
        la, rb = s.profiles[m.left.dataset_id].column(m.left.column), s.profiles[m.right.dataset_id].column(m.right.column)
        temporal = {"DATE", "DATETIME"}
        if {ds[m.left.dataset_id], ds[m.right.dataset_id]} == {"rev", "orders"}:
            assert not (la.semantic_type.value in temporal and rb.semantic_type.value in temporal), f"{fmt}: unrelated dates linked: {m.left.key} ↔ {m.right.key}"
        if la.semantic_type.value == "ID" and rb.semantic_type.value == "ID":
            assert max(m.evidence.containment_left, m.evidence.containment_right) >= 0.05, f"{fmt}: disjoint identifiers linked {m.left.key} ↔ {m.right.key}"
    # the schema map is built for any datasets and places every table
    view = s.graph_view("schema")
    assert len(view["tables"]) == 6 and all("layer" in t for t in view["tables"])
    rev_orders = next(r for r in view["relationships"] if {ds[r["from"]], ds[r["to"]]} == {"rev", "orders"})
    assert ds[rev_orders["from"]] == "rev" and rev_orders["cardinality"] == "N : 1"


def test_row_verification_keeps_true_and_rejects_false_attribute(tmp_path):
    """A denormalised copy of a parent attribute agrees on joined rows (kept); a look-alike does not (rejected)."""
    rng = random.Random(9)
    n = 800
    keys = [f"K{i:05d}" for i in range(n)]
    region = [rng.choice(["north", "south", "east", "west"]) for _ in keys]
    d0 = date(2022, 1, 1)
    opened = [d0 + timedelta(days=rng.randrange(400)) for _ in keys]
    parent = pd.DataFrame({"account_ref": keys, "sales_region": region, "opened_on": opened})
    rows = [rng.randrange(n) for _ in range(3000)]
    child = pd.DataFrame({
        "acct": [keys[i] for i in rows],
        "region_name": [region[i] for i in rows],                                        # same information (denormalised)
        "opened_dt": [d0 + timedelta(days=rng.randrange(400)) for _ in rows],            # looks identical, different per row
        "qty": [rng.randint(1, 9) for _ in rows],
    })
    (tmp_path / "d").mkdir()
    parent.to_csv(tmp_path / "d" / "accounts.csv", index=False)
    child.to_csv(tmp_path / "d" / "events.csv", index=False)
    s = Session(workspace_root=tmp_path / "ws")
    for f in sorted((tmp_path / "d").glob("*.csv")):
        s.add_file(f)
    s.discover()
    by_cols = {frozenset((m.left.column, m.right.column)): m for m in s.matching.matches}
    region_m = by_cols.get(frozenset(("sales_region", "region_name")))
    date_m = by_cols.get(frozenset(("opened_on", "opened_dt")))
    assert region_m is not None and region_m.accepted and (region_m.evidence.row_agreement or 0) > 0.95
    assert date_m is None or not date_m.accepted, "look-alike date column must not be accepted"
    if date_m is not None and date_m.evidence.row_agreement is not None:
        assert date_m.evidence.row_agreement < 0.5


def test_same_file_loaded_twice_merges_without_measure_keys(tmp_path, sample_files):
    """Uploading overlapping files (here the same one twice) must not join on money columns or crash the merge."""
    s = Session(workspace_root=tmp_path / "ws")
    sales = next(f for f in sample_files if f.name.endswith(".parquet"))
    for f in sample_files:
        s.add_file(f)
    s.add_file(sales)  # second copy → sample_sales_2
    s.discover()
    for r in s.relationships:
        for m in r.key_matches:
            for ref in (m.left, m.right):
                assert s.profiles[ref.dataset_id].column(ref.column).semantic_type.value not in ("CURRENCY", "FREE_TEXT", "BOOLEAN"), f"measure used as join key: {ref.key}"
    rec = s.execute(write_csv=False)
    assert rec.result.validation["passed"]
