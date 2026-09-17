"""Graph experiment: direct matching versus graph-assisted integration routes.

Controlled scenario (known ground truth):

    A  orders    order_id, buyer_account ('0000123'), buyer_label ('R. Sharma'), amount
    B  accounts  account_no (123), holder_name, email
    C  contacts  contact_email, contact_name, region            (no identifier)

    A ── strong key (buyer_account ≡ account_no under identifier normalisation) ── B
    B ── strong entity evidence (e-mail + name) ────────────────────────────────── C
    A ── only a weak, abbreviated name ────────────────────────────────────────── C

Task: attach the right contact (C) to every order (A).

* direct matching   — link A to C using only what they share (abbreviated names);
* graph-assisted    — the system discovers the route A → B → C with Dijkstra on
                      −log confidence and integrates along the maximum spanning tree.

The real-world counterpart is reported too: in the NYC scenario, trips have no
direct relationship to census data; the route trips → zones → NTA → census is found.

    python experiments/run_graph_experiment.py
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from common import RESULTS, ROOT, grouped_bars, markdown_table, save_json

from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.resolver import FieldSpec, resolve
from backend.graph import algorithms as ga
from backend.session.state import Session

FIRST = ["Rahul", "Rohan", "Riya", "Priya", "Pranav", "Amit", "Anita", "Arjun", "Kavya", "Karan", "Meera", "Mohan", "Neha", "Nikhil", "Sara", "Sameer"]
LAST = ["Sharma", "Shah", "Iyer", "Rao", "Nair", "Mehta", "Gupta", "Kapoor", "Das", "Reddy"]


def generate(out: Path, n_people: int = 800, n_orders: int = 4000, seed: int = 5) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    people = []
    for i in range(n_people):
        f, l = rng.choice(FIRST), rng.choice(LAST)
        people.append({"entity": i, "first": f, "last": l, "email": f"{f.lower()}.{l.lower()}.{i}@example.org", "account": 1000 + i})
    # B: accounts
    b_rows = rng.sample(people, int(n_people * 0.95))
    pq.write_table(pa.table({"account_no": [p["account"] for p in b_rows], "holder_name": [f"{p['first']} {p['last']}" for p in b_rows], "email": [p["email"] for p in b_rows]}), out / "accounts.parquet")
    # C: contacts (different system: no ids, some name variants)
    c_rows = rng.sample(people, int(n_people * 0.9))
    names = [f"{p['first']} {p['last']}" if rng.random() > 0.3 else f"{p['first'].upper()} {p['last'].upper()}" for p in c_rows]
    pq.write_table(pa.table({"contact_email": [p["email"] for p in c_rows], "contact_name": names, "region": [rng.choice(["north", "south", "east", "west"]) for _ in c_rows]}), out / "contacts.parquet")
    # A: orders reference accounts by zero-padded code and carry only an abbreviated buyer label
    a_ent = []
    order_rows = {"order_id": [], "buyer_account": [], "buyer_label": [], "amount": []}
    for i in range(n_orders):
        p = rng.choice(people)
        a_ent.append(p["entity"])
        order_rows["order_id"].append(f"O{i:06d}")
        order_rows["buyer_account"].append(f"{p['account']:07d}")
        order_rows["buyer_label"].append(f"{p['first'][0]}. {p['last']}")
        order_rows["amount"].append(round(rng.uniform(5, 500), 2))
    pq.write_table(pa.table(order_rows), out / "orders.parquet")
    return {"orders": a_ent, "accounts": [p["entity"] for p in b_rows], "contacts": [p["entity"] for p in c_rows]}


def accuracy(links: dict[int, set[int]], gt: dict, n_orders: int) -> dict:
    """links: order row -> set of contact rows assigned. Correct iff exactly the true contact (when one exists)."""
    contact_of = defaultdict(set)
    for row, ent in enumerate(gt["contacts"]):
        contact_of[ent].add(row)
    linkable = sum(1 for e in gt["orders"] if contact_of[e])
    correct = wrong = 0
    for i in range(n_orders):
        got = links.get(i, set())
        if not got:
            continue
        if got == contact_of[gt["orders"][i]]:
            correct += 1
        else:
            wrong += 1
    return {"linkable_orders": linkable, "linked": correct + wrong, "correct": correct, "wrong": wrong,
            "precision": round(correct / (correct + wrong), 4) if correct + wrong else 0.0, "recall": round(correct / linkable, 4) if linkable else 0.0}


def main() -> None:
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_graphexp_"))
    try:
        gt = generate(tmp / "data")
        n_orders = len(gt["orders"])
        files = {k: tmp / "data" / f"{k}.parquet" for k in ("orders", "accounts", "contacts")}

        # ---------------------------------------------------------------- graph-assisted (full system)
        s = Session(workspace_root=tmp / "ws")
        for f in files.values():
            s.add_file(f)
        s.discover()
        rels = [{"left": r.left_dataset, "right": r.right_dataset, "join_kind": r.join_kind.value, "confidence": r.confidence} for r in s.relationships]
        route = s.route("orders", "contacts")
        route_linear = s.route("orders", "contacts", cost_mode="linear")
        rec = s.execute(write_csv=False)
        out = rec.result.output_path.replace("\\", "/")
        ent_file = next(v for k, v in rec.files.items() if k.startswith("entities_")).replace("\\", "/")
        members = {}
        for eid, sources in duckdb.sql(f"SELECT entity_id, _entity_sources FROM read_parquet('{ent_file}')").fetchall():
            members[eid] = {int(t.rsplit(":", 1)[1]) for t in json.loads(sources) if t.startswith("contacts:")}
        links = {}
        ent_col = next(c for c in rec.result.columns if c == "entity_id" or c.endswith("_entity_id"))
        for row, eid in duckdb.sql(f"SELECT _src_orders_row, \"{ent_col}\" FROM read_parquet('{out}')").fetchall():
            if eid is not None and members.get(eid):
                links[row] = members[eid]
        graph_acc = accuracy(links, gt, n_orders)

        # ---------------------------------------------------------------- direct matching A ↔ C
        s_direct = Session(workspace_root=tmp / "ws_direct")
        for k in ("orders", "contacts"):
            s_direct.add_file(files[k])
        direct_summary = s_direct.discover()
        direct_rel = direct_summary["relationships"]
        a, c = s_direct.artifacts["orders"], s_direct.artifacts["contacts"]
        er = resolve({"orders": a.storage_path, "contacts": c.storage_path}, [FieldSpec("name", S.NAME, {"orders": "buyer_label", "contacts": "contact_name"})], cfg,
                     dedupe=set(), auto_threshold=0.5, review_threshold=0.3)
        by_cluster = defaultdict(lambda: {"orders": set(), "contacts": set()})
        for (ds, row), eid in er.assignment.items():
            by_cluster[eid][ds].add(row)
        direct_links = {o: g["contacts"] for g in by_cluster.values() for o in g["orders"] if g["contacts"]}
        direct_acc = accuracy(direct_links, gt, n_orders)

        # ---------------------------------------------------------------- NYC real-world counterpart
        nyc = None
        nyc_dir = ROOT / "data/processed/nyc"
        if (nyc_dir / "yellow_tripdata_sample.parquet").exists():
            sn = Session(workspace_root=tmp / "ws_nyc")
            for f in ("yellow_tripdata_sample.parquet", "taxi_zone_lookup.csv", "nta_demographics.json", "census_acs_nyc_counties.json"):
                sn.add_file(nyc_dir / f)
            sn.discover()
            r = sn.route("yellow_tripdata_sample", "census_acs_nyc_counties")
            nyc = {"path": r["path"], "reliability": r["reliability"], "direct_confidence": r["direct_confidence"], "hops": r["hops"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    result = {
        "relationships_discovered": rels,
        "dijkstra_route_neglog": {k: route[k] for k in ("path", "reliability", "total_cost", "direct_confidence", "alternatives")},
        "dijkstra_route_linear": {k: route_linear[k] for k in ("path", "reliability", "total_cost", "direct_confidence")},
        "graph_assisted": graph_acc,
        "direct_matching": {**direct_acc, "direct_relationship_found": direct_rel},
        "merge_plan_excluded": rec.plan.excluded_relationships,
        "nyc_route": nyc,
    }
    save_json("graph_experiment.json", result)
    rows = [{"approach": "Direct A↔C (shared abbreviated names)", **direct_acc}, {"approach": "Graph route A→B→C (system)", **graph_acc}]
    md = ["# Graph experiment: direct matching vs graph-assisted routes", "",
          "Relationships discovered by the system:", "",
          markdown_table(rels, ["left", "right", "join_kind", "confidence"]), "",
          f"Dijkstra (cost = −log c): {' → '.join(route['path'])}, reliability {route['reliability']:.3f}; direct A–C confidence: {route['direct_confidence']}.",
          f"Linear cost (1 − c): {' → '.join(route_linear['path'])}, reliability {route_linear['reliability']:.3f}.", "",
          "Linking each order to the correct contact:", "",
          markdown_table(rows, ["approach", "linkable_orders", "linked", "correct", "wrong", "precision", "recall"], fmt={"precision": ".3f", "recall": ".3f"}), ""]
    if rec.plan.excluded_relationships:
        md += ["Relationships excluded by the maximum spanning tree:", ""] + [f"- {e['left']} ↔ {e['right']} ({e['confidence']:.2f}): {e['reason']}" for e in rec.plan.excluded_relationships] + [""]
    if nyc:
        md += ["## Real-world counterpart (NYC)", "", f"No direct trips ↔ census relationship exists (direct confidence: {nyc['direct_confidence']}). Route found: {' → '.join(nyc['path'])} (reliability {nyc['reliability']:.3f}).", ""]
        md += [f"- {h['from']} → {h['to']}: {h['join_kind']} ({h['confidence']:.2f}) via {', '.join(h['keys'])}" for h in nyc["hops"]]
    (RESULTS / "graph_experiment.md").write_text("\n".join(md), encoding="utf-8")
    grouped_bars(RESULTS / "graph_experiment.png", ["precision", "recall"], {"Direct A↔C": [direct_acc["precision"], direct_acc["recall"]], "Graph route A→B→C": [graph_acc["precision"], graph_acc["recall"]]},
                 "Linking orders to contacts: direct vs graph-assisted", "score", ylim=(0, 1.12))
    print("\n".join(md))


if __name__ == "__main__":
    main()
