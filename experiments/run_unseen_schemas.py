"""Relationship inference and merge planning on public schemas the system was never tuned on.

Each database is loaded (every table), discovered, planned and merged with default settings. The
published foreign keys are the ground truth and are used only here, for scoring:

* relationships: a published FK counts as found when the two tables are linked on those columns with
  the right direction (a lookup whose dimension is the referenced table, or a same-entity key merge);
  links between tables without a published FK are counted as extra;
* root: the merge should keep the grain of a fact table;
* per-join match rate: compared with the rate computed with DuckDB on the raw tables.

    python scripts/download_unseen_schemas.py   # Northwind CSVs + Chinook SQLite -> data/raw/
    python experiments/run_unseen_schemas.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import duckdb

from common import RESULTS, ROOT, markdown_table, save_json

from backend.core.models import JoinKind
from backend.session.state import Session

RAW = ROOT / "data" / "raw"

# (child table, child column, parent table, parent column): published foreign keys, self-references omitted
NORTHWIND = {
    "files": {t: RAW / "northwind" / f"{t}.csv" for t in ["categories", "customers", "employees", "order-details", "orders", "products", "shippers", "suppliers"]},
    "fks": [("order-details", "orderID", "orders", "orderID"), ("order-details", "productID", "products", "productID"),
            ("orders", "customerID", "customers", "customerID"), ("orders", "employeeID", "employees", "employeeID"),
            ("orders", "shipVia", "shippers", "shipperID"), ("products", "categoryID", "categories", "categoryID"),
            ("products", "supplierID", "suppliers", "supplierID")],
    "facts": ["order-details"],
}
CHINOOK = {
    "sqlite": RAW / "chinook" / "Chinook_Sqlite.sqlite",
    "fks": [("Album", "ArtistId", "Artist", "ArtistId"), ("Track", "AlbumId", "Album", "AlbumId"), ("Track", "MediaTypeId", "MediaType", "MediaTypeId"),
            ("Track", "GenreId", "Genre", "GenreId"), ("InvoiceLine", "InvoiceId", "Invoice", "InvoiceId"), ("InvoiceLine", "TrackId", "Track", "TrackId"),
            ("Invoice", "CustomerId", "Customer", "CustomerId"), ("Customer", "SupportRepId", "Employee", "EmployeeId"),
            ("PlaylistTrack", "PlaylistId", "Playlist", "PlaylistId"), ("PlaylistTrack", "TrackId", "Track", "TrackId")],
    "facts": ["InvoiceLine", "PlaylistTrack"],
}
OLIST = {
    "files": {p.stem.replace("olist_", "").replace("_dataset", ""): p for p in sorted((RAW / "olist").glob("*.csv"))},
    "fks": [("order_items", "order_id", "orders", "order_id"), ("order_items", "product_id", "products", "product_id"),
            ("order_items", "seller_id", "sellers", "seller_id"), ("order_payments", "order_id", "orders", "order_id"),
            ("order_reviews", "order_id", "orders", "order_id"), ("orders", "customer_id", "customers", "customer_id"),
            ("products", "product_category_name", "product_category_name_translation", "product_category_name")],
    "facts": ["order_items"],
}


def load(spec: dict, ws: Path) -> tuple[Session, dict[str, str], dict[str, str]]:
    """The session, table -> dataset id, and table -> DuckDB source of the ingested copy (for ground truth)."""
    s = Session(workspace_root=ws)
    if "sqlite" in spec:
        named = {a.metadata["table"]: a for a in s.add_files(spec["sqlite"])}  # every table of the database
    else:
        named = {t: s.add_file(p) for t, p in spec["files"].items()}
    return s, {t: a.dataset_id for t, a in named.items()}, {t: f"'{a.storage_path}'" for t, a in named.items()}


def evaluate(spec: dict, ws: Path) -> dict:
    t0 = time.perf_counter()
    s, ids, src = load(spec, ws)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    s.discover()
    discover_s = time.perf_counter() - t0
    table = {v: k for k, v in ids.items()}

    found, links = set(), []
    for r in s.relationships:
        cols = {(table[m.left.dataset_id], m.left.column, table[m.right.dataset_id], m.right.column) for m in r.key_matches}
        dim = table.get(r.evidence.get("dimension"), "")
        links.append({"left": table[r.left_dataset], "right": table[r.right_dataset], "kind": r.join_kind.value, "confidence": round(r.confidence, 3),
                      "dimension": dim, "keys": sorted(f"{a}.{b} = {c}.{d}" for a, b, c, d in cols)})
        for fk in spec["fks"]:
            child, ccol, parent, pcol = fk
            if cols & {(child, ccol, parent, pcol), (parent, pcol, child, ccol)} and (
                    r.join_kind == JoinKind.ENTITY_KEY_MERGE or (r.join_kind == JoinKind.LOOKUP and dim == parent)):
                found.add(fk)
    fk_pairs = {frozenset((c, p)) for c, _, p, _ in spec["fks"]}
    extra = [l for l in links if frozenset((l["left"], l["right"])) not in fk_pairs]
    missed = [f"{c}.{cc} -> {p}.{pc}" for c, cc, p, pc in spec["fks"] if (c, cc, p, pc) not in found]

    root = table[s.make_plan().root_dataset]
    t0 = time.perf_counter()
    rec = s.execute(write_csv=False)
    merge_s = time.perf_counter() - t0
    root_rows = duckdb.sql(f"SELECT count(*) FROM {src[root]}").fetchone()[0]

    joins = []
    for j in rec.join_stats:
        p, c = table[j["parent"]], table[j["child"]]
        truth = None
        if j["kind"] == "lookup" and len(j["keys"]) == 1:
            k = j["keys"][0]
            # share of parent rows whose (non-null) key value exists in the child table
            truth = round(float(duckdb.sql(
                f"""SELECT avg(CASE WHEN c.k IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM
                    (SELECT CAST("{k['parent_column']}" AS VARCHAR) k FROM {src[p]}) x
                    LEFT JOIN (SELECT DISTINCT CAST("{k['child_column']}" AS VARCHAR) k FROM {src[c]}) c ON x.k = c.k""").fetchone()[0]), 4)
        joins.append({"join": f"{p} <- {c}", "kind": j["kind"], "reported": j["match_rate"], "truth": truth,
                      "ok": None if truth is None else abs(j["match_rate"] - truth) < 0.005})
    return {
        "tables": len(ids), "rows": sum(duckdb.sql(f"SELECT count(*) FROM {v}").fetchone()[0] for v in src.values()),
        "published_fks": len(spec["fks"]), "found": len(found), "missed": missed, "extra_links": extra,
        "root": root, "root_is_fact": root in spec["facts"], "output_rows": rec.result.row_count, "root_rows": root_rows,
        "grain_kept": rec.result.row_count == root_rows, "validation": rec.result.validation.get("passed"),
        "joins": joins, "links": links, "seconds": {"load": round(load_s, 1), "discover": round(discover_s, 1), "merge": round(merge_s, 1)},
    }


def main() -> None:
    out = []
    for name, spec in [("Northwind", NORTHWIND), ("Chinook", CHINOOK), ("Olist", OLIST)]:
        paths = [spec["sqlite"]] if "sqlite" in spec else list(spec["files"].values())
        if not paths or not all(Path(p).exists() for p in paths):
            print(f"skip {name}: data missing (python scripts/download_unseen_schemas.py / scripts/download_olist.py)")
            continue
        r = {"database": name, **evaluate(spec, Path(tempfile.mkdtemp(prefix=f"dfg_unseen_{name}_")))}
        out.append(r)
        print(f"\n== {name}: FKs {r['found']}/{r['published_fks']}, extra {len(r['extra_links'])}, root {r['root']} (fact: {r['root_is_fact']}), "
              f"rows {r['output_rows']} vs root {r['root_rows']}, validation {r['validation']}, {r['seconds']}")
        for m in r["missed"]:
            print("   missed:", m)
        for l in r["extra_links"]:
            print("   extra :", l["left"], "-", l["right"], l["kind"], l["confidence"], l["keys"])
        for l in r["links"]:
            print("   link  :", l["left"], "-", l["right"], l["kind"], "dim", l["dimension"], l["confidence"], l["keys"])
        for j in r["joins"]:
            flag = "" if j["ok"] in (None, True) else "MISMATCH"
            print(f"   join  : {j['join']:48s} {j['kind']:18s} reported {j['reported']:.4f}  truth {j['truth']}  {flag}")
    save_json("unseen_schemas.json", out)
    rows = [{"database": r["database"], "tables": r["tables"], "rows": r["rows"], "fks": f"{r['found']}/{r['published_fks']}", "extra": len(r["extra_links"]),
             "root": r["root"], "root_is_fact": "yes" if r["root_is_fact"] else "no", "grain": "yes" if r["grain_kept"] else "no",
             "rates": f"{sum(1 for j in r['joins'] if j['ok'])}/{sum(1 for j in r['joins'] if j['ok'] is not None)}",
             "validation": {True: "passed", False: "failed", None: "-"}[r["validation"]], "seconds": sum(r["seconds"].values())} for r in out]
    md = ["# Unseen public schemas", "", "Published foreign keys are the ground truth; nothing in the system was tuned on these databases.", "",
          markdown_table(rows, ["database", "tables", "rows", "fks", "extra", "root", "root_is_fact", "grain", "rates", "validation", "seconds"],
                         ["database", "tables", "rows", "published FKs found", "extra links", "merge root", "root is a fact table", "grain kept",
                          "lookup match rates correct", "validation", "seconds"], {"seconds": ".1f"})]
    for r in out:
        md += ["", f"## {r['database']}", "", f"* missed: {', '.join(r['missed']) or 'none'}",
               f"* extra links: {', '.join(l['left'] + ' - ' + l['right'] + ' (' + l['kind'] + ')' for l in r['extra_links']) or 'none'}"]
    (RESULTS / "unseen_schemas.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
