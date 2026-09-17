"""Entity-resolution and end-to-end integration evaluation.

Part 1 — entity resolution on the sample scenario (3 generator seeds), pairwise
precision / recall / F1 against the hidden entity ids:

    exact_email          link records with identical normalised e-mail (deterministic baseline)
    fuzzy_name_city      Jaro-Winkler(name) >= 0.90 and same city, transitive closure
    fs_components        Fellegi-Sunter probabilities + connected components
    fs_correlation       Fellegi-Sunter + correlation clustering (the system)

Part 2 — end-to-end integration (full session pipeline) versus a naive
exact-column-name join: match coverage, unmatched ratio, conflict rate,
duplicate reduction, integration accuracy (entity purity of the customer
attached to each transaction), runtime and peak memory.

    python experiments/run_evaluation.py [--seeds 7 11 23]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import networkx as nx
from rapidfuzz.distance import JaroWinkler

from common import RESULTS, PeakMemory, grouped_bars, markdown_table, mean, prf, save_json

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.core.values import basic_normalize, get_normalizer
from backend.entity_resolution.comparators import normalize_person_name
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve
from backend.ingestion.loader import ingest_file
from backend.session.state import Session
from backend.storage.workspace import Workspace


def truth_pairs(entities: dict[str, list], datasets: list[str]) -> set:
    groups = defaultdict(list)
    for ds in datasets:
        for row, ent in enumerate(entities[ds]):
            if ent is not None:
                groups[ent].append((ds, row))
    return {tuple(sorted(p)) for g in groups.values() for p in itertools.combinations(g, 2)}


def cluster_pairs(groups) -> set:
    return {tuple(sorted(p)) for g in groups for p in itertools.combinations(sorted(g), 2)}


def er_part(seed: int, tmp: Path, cfg: dict) -> list[dict]:
    d = tmp / f"seed{seed}"
    make_sample_data.generate(d, seed=seed)
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    ws = Workspace(tmp / "ws", f"er{seed}")
    a = ingest_file(ws, d / "sample_customers.csv")
    b = ingest_file(ws, d / "sample_customer_master.json")
    ids = {a.dataset_id: "sample_customers", b.dataset_id: "sample_customer_master"}
    fields = [
        FieldSpec("name", S.NAME, {a.dataset_id: "customer_name", b.dataset_id: "full_name"}),
        FieldSpec("city", S.CITY, {a.dataset_id: "city", b.dataset_id: "location"}, "semantic:CITY"),
        FieldSpec("email", S.EMAIL, {a.dataset_id: "email", b.dataset_id: "email_address"}),
        FieldSpec("phone", S.PHONE, {a.dataset_id: "phone", b.dataset_id: "phone_number"}, "digits"),
    ]
    paths = {a.dataset_id: a.storage_path, b.dataset_id: b.storage_path}
    truth = truth_pairs({k: gt["entities"][v] for k, v in ids.items()}, list(ids))
    records = load_records(paths, fields)
    names = list(paths)
    nodes = [(names[di], r) for di, r in records]
    out = []

    t0 = time.perf_counter()
    g = nx.Graph()
    g.add_nodes_from(nodes)
    by_email = defaultdict(list)
    for (di, r), rec in records.items():
        e = basic_normalize(rec.get("email"))
        if e:
            by_email[e].append((names[di], r))
    for members in by_email.values():
        g.add_edges_from(zip(members, members[1:]))
    out.append({"method": "exact_email", "seed": seed, **prf(cluster_pairs(nx.connected_components(g)), truth), "seconds": round(time.perf_counter() - t0, 3)})

    t0 = time.perf_counter()
    city = get_normalizer("semantic:CITY")
    g = nx.Graph()
    g.add_nodes_from(nodes)
    blocks = defaultdict(list)
    for (di, r), rec in records.items():
        n = normalize_person_name(rec.get("name"))
        if n:
            blocks[(n.split()[-1], city(rec.get("city")))].append(((names[di], r), n))
    for members in blocks.values():
        for (x, nx_), (y, ny_) in itertools.combinations(members, 2):
            if JaroWinkler.similarity(nx_, ny_) >= 0.9:
                g.add_edge(x, y)
    out.append({"method": "fuzzy_name_city", "seed": seed, **prf(cluster_pairs(nx.connected_components(g)), truth), "seconds": round(time.perf_counter() - t0, 3)})

    old_prior = copy.deepcopy(cfg)
    old_prior["entity_resolution"]["prior_population"] = "candidates"  # previous default: overconfident probabilities
    for method, clustering, thr, mcfg in (("fs_correlation_oldprior", "correlation", 0.9, old_prior),
                                          ("fs_components", "connected_components", 0.9, cfg), ("fs_correlation", "correlation", 0.9, cfg),
                                          ("fs_components@0.5", "connected_components", 0.5, cfg), ("fs_correlation@0.5", "correlation", 0.5, cfg)):
        t0 = time.perf_counter()
        res = resolve(paths, fields, mcfg, auto_threshold=thr, clustering=clustering, records=records)
        groups = defaultdict(list)
        for rec, ent in res.assignment.items():
            groups[ent].append(rec)
        out.append({"method": method, "seed": seed, **prf(cluster_pairs(groups.values()), truth), "seconds": round(time.perf_counter() - t0, 3),
                    "reduction_ratio": res.stats["reduction_ratio"], "candidate_pairs": res.stats["candidate_pairs"]})
    return out


def integration_part(seed: int, tmp: Path) -> dict:
    d = tmp / f"seed{seed}"
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    files = [d / "sample_customers.csv", d / "sample_customer_master.json", d / "sample_sales.parquet"]
    started = time.perf_counter()
    with PeakMemory() as mem:
        s = Session(workspace_root=tmp / f"int{seed}")
        for f in files:
            s.add_file(f)
        rec = s.execute(write_csv=False)
    seconds = time.perf_counter() - started
    q = rec.result.quality_report
    out = rec.result.output_path.replace("\\", "/")
    rows = duckdb.sql(f"SELECT _src_sample_sales_row, _src_sample_customers_row, entity_id FROM read_parquet('{out}')").fetchall()

    # integration accuracy: does the entity attached to each transaction hold only records of the transaction's true customer?
    er = next(iter(rec.group_stats.values()))
    members_by_entity = defaultdict(list)
    ent_tables = rec.files[f"entities_{next(iter(rec.group_stats))}.parquet"].replace("\\", "/")
    for eid, sources in duckdb.sql(f"SELECT entity_id, _entity_sources FROM read_parquet('{ent_tables}')").fetchall():
        for token in json.loads(sources):
            ds, row = token.rsplit(":", 1)
            members_by_entity[eid].append((ds, int(row)))
    truth_of = {"sample_customers": gt["entities"]["sample_customers"], "sample_customer_master": gt["entities"]["sample_customer_master"]}
    correct = total = 0
    for sales_row, _cust_row, eid in rows:
        true_ent = gt["entities"]["sample_sales"][sales_row]
        if true_ent is None or eid is None:
            continue
        total += 1
        members = [truth_of[ds][r] for ds, r in members_by_entity[eid]]
        correct += int(Counter(members).most_common(1)[0][0] == true_ent and all(m == true_ent for m in members))
    true_dups = gt["stats"]["crm_duplicate_accounts"] + gt["stats"]["crm_exact_duplicate_rows"]
    fused_cells = er["entities"] * 4
    system = {
        "seed": seed, "system": "GraphFusion",
        "match_coverage": q["integration_coverage"], "row_match_rate": round(rec.result.matched_rows / rec.result.row_count, 4),
        "unmatched_ratio": round(rec.result.unmatched_rows / rec.result.row_count, 4),
        "conflict_rate": round(rec.result.conflicts / fused_cells, 4),
        "duplicate_reduction": round(min(1.0, er["within_dataset_duplicates_resolved"] / true_dups), 4),
        "integration_accuracy": round(correct / total, 4) if total else 0.0,
        "true_orphan_rate": round(gt["stats"]["sales_orphans"] / gt["stats"]["sales_rows"], 4),
        "runtime_s": round(seconds, 2), "peak_rss_mb": mem.peak_mb,
    }
    # naive baseline: join only on identical column names (the pandas.concat/merge-on-common-columns approach)
    con = duckdb.connect()
    cols = {f.stem: {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{Path(s.artifacts[f.stem].storage_path).as_posix()}')").fetchall()} for f in files}
    common = cols["sample_sales"] & cols["sample_customers"]
    naive = {"seed": seed, "system": "naive exact-name join", "match_coverage": 0.0 if not common else None, "row_match_rate": 0.0 if not common else None,
             "unmatched_ratio": 1.0 if not common else None, "conflict_rate": 0.0, "duplicate_reduction": 0.0, "integration_accuracy": 0.0, "true_orphan_rate": system["true_orphan_rate"],
             "runtime_s": 0.0, "peak_rss_mb": None, "note": f"shared column names between sales and customers: {sorted(common) or 'none'}"}
    return {"system": system, "naive": naive}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    args = ap.parse_args()
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_eval_"))
    er_rows, integ = [], []
    try:
        for seed in args.seeds:
            er_rows += er_part(seed, tmp, cfg)
            integ.append(integration_part(seed, tmp))
            print(f"  seed {seed} done")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    methods = ["exact_email", "fuzzy_name_city", "fs_correlation_oldprior", "fs_components", "fs_correlation", "fs_components@0.5", "fs_correlation@0.5"]
    er_summary = [{"method": m, **{k: round(mean(r[k] for r in er_rows if r["method"] == m), 4) for k in ("precision", "recall", "f1", "seconds")}} for m in methods]
    sys_rows = [x["system"] for x in integ]
    int_summary = {k: round(mean(r[k] for r in sys_rows), 4) for k in ("match_coverage", "row_match_rate", "unmatched_ratio", "conflict_rate", "duplicate_reduction", "integration_accuracy", "true_orphan_rate", "runtime_s", "peak_rss_mb")}
    save_json("evaluation.json", {"entity_resolution": {"per_seed": er_rows, "summary": er_summary}, "integration": {"per_seed": integ, "summary": int_summary}})

    labels = {"exact_email": "Exact e-mail", "fuzzy_name_city": "Fuzzy name+city", "fs_correlation_oldprior": "FS old prior + correlation (t=0.9)", "fs_components": "FS + components (t=0.9)", "fs_correlation": "FS + correlation (t=0.9)", "fs_components@0.5": "FS + components (t=0.5)", "fs_correlation@0.5": "FS + correlation (t=0.5)"}
    md = ["# Entity resolution and integration evaluation", "", f"Seeds: {args.seeds} (600 entities, ~1,000 customer records per seed).", "", "## Entity resolution (pairwise, mean over seeds)", "",
          markdown_table([{**r, "label": labels[r["method"]]} for r in er_summary], ["label", "precision", "recall", "f1", "seconds"], ["method", "precision", "recall", "F1", "time (s)"], {"precision": ".3f", "recall": ".3f", "f1": ".3f", "seconds": ".2f"}),
          "", "## End-to-end integration (mean over seeds)", "",
          markdown_table([{"metric": k, "GraphFusion": v, "naive exact-name join": {"match_coverage": 0.0, "row_match_rate": 0.0, "unmatched_ratio": 1.0, "conflict_rate": "n/a", "duplicate_reduction": 0.0, "integration_accuracy": 0.0, "true_orphan_rate": int_summary["true_orphan_rate"], "runtime_s": "—", "peak_rss_mb": "—"}[k]} for k, v in int_summary.items()],
                         ["metric", "GraphFusion", "naive exact-name join"]),
          "", f"Naive baseline: {integ[0]['naive']['note']} — without schema matching nothing can be joined.",
          "", "Notes: `row_match_rate` is bounded by the true orphan rate (transactions referencing customers that do not exist). `conflict_rate` = conflicting fused attribute values / fused attribute cells."]
    (RESULTS / "evaluation.md").write_text("\n".join(md), encoding="utf-8")
    grouped_bars(RESULTS / "entity_resolution.png", ["precision", "recall", "F1"], {labels[r["method"]]: [r["precision"], r["recall"], r["f1"]] for r in er_summary}, "Entity resolution on the sample scenario (mean of 3 seeds)", "score", ylim=(0, 1.12))
    print("\n".join(md))


if __name__ == "__main__":
    main()
