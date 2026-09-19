"""Root choice on star schemas and per-join match statistics with nested lookups."""
import random
import uuid

import pandas as pd

from backend.session.state import Session


def _star(tmp_path):
    rng = random.Random(5)
    hexid = lambda: uuid.UUID(int=rng.getrandbits(128)).hex  # noqa: E731
    zips = [f"{rng.randint(10000, 99999)}" for _ in range(40)]
    cats = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    orders = pd.DataFrame({"order_id": [hexid() for _ in range(400)], "order_status": [rng.choice(["new", "paid", "sent"]) for _ in range(400)],
                           "buyer_zip": [rng.choice(zips) for _ in range(400)]})
    products = pd.DataFrame({"product_id": [hexid() for _ in range(120)], "product_category": [rng.choice(cats) for _ in range(120)],
                             "weight_g": [rng.randint(100, 5000) for _ in range(120)]})
    items = pd.DataFrame({"order_id": [rng.choice(orders.order_id) for _ in range(900)], "product_id": [rng.choice(products.product_id) for _ in range(900)],
                          "price": [round(rng.uniform(5, 500), 2) for _ in range(900)]})
    # translation covers only 4 of 6 categories: the nested lookup is partial, the products lookup is complete
    translation = pd.DataFrame({"product_category": cats[:4], "product_category_english": ["A", "B", "G", "D"]})
    # a large reference table keyed by a non-unique zip: it can only join through aggregation
    geo = pd.DataFrame({"zip_prefix": [rng.choice(zips) for _ in range(5000)], "lat": [rng.uniform(-30, 0) for _ in range(5000)],
                        "lng": [rng.uniform(-60, -30) for _ in range(5000)]})
    paths = []
    for name, df in {"orders": orders, "products": products, "items": items, "translation": translation, "geo": geo}.items():
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths, items


def test_root_is_the_fact_table_not_the_largest_aggregated_reference(tmp_path):
    paths, items = _star(tmp_path)
    s = Session(workspace_root=tmp_path / "ws")
    for p in paths:
        s.add_file(p)
    plan = s.make_plan()
    assert plan.root_dataset == "items", plan.root_dataset  # geo has 5.5x more rows but only joins through aggregation
    rec = s.execute(write_csv=False)
    assert rec.result.row_count == len(items)


def test_lookup_match_rate_is_its_own_not_a_nested_joins(tmp_path):
    paths, items = _star(tmp_path)
    s = Session(workspace_root=tmp_path / "ws")
    for p in paths:
        s.add_file(p)
    rec = s.execute(write_csv=False)
    stats = {(j["parent"], j["child"]): j for j in rec.join_stats}
    prod = stats[("items", "products")]
    assert prod["matched_rows"] == len(items), prod  # every item's product exists
    trans = stats[("products", "translation")]
    assert 0 < trans["unmatched_rows"] < trans["rows"]  # two categories have no translation
