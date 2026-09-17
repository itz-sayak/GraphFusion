"""How far does single-machine entity resolution scale?

Generates the customer scenario at increasing sizes (two customer systems, ~1.7 records per entity)
and runs blocking + Fellegi–Sunter + correlation clustering in one process. Reports records,
candidate pairs, wall time, peak memory and pairwise F1. The trend decides whether distributed ER
(Spark/Ray) is needed for this system's entity tables.

    python experiments/run_er_scaling.py [--sizes 2000 10000 40000]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from common import RESULTS, PeakMemory, markdown_table, prf, save_json
from run_evaluation import cluster_pairs, truth_pairs

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve
from backend.ingestion.loader import ingest_file
from backend.storage.workspace import Workspace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 40000, 100000])
    ap.add_argument("--no-refine", action="store_true", help="disable adaptive block refinement (baseline)")
    args = ap.parse_args()
    cfg = load_config()
    if args.no_refine:
        cfg["entity_resolution"]["refine_block_size"] = None
    suffix = "_norefine" if args.no_refine else ""
    tmp = Path(tempfile.mkdtemp(prefix="dfg_scale_"))
    rows = []
    for n in args.sizes:
        d = tmp / f"n{n}"
        make_sample_data.generate(d, n_entities=n, n_sales=10, seed=7)
        gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
        ws = Workspace(tmp / "ws", f"s{n}")
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
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            records = load_records(paths, fields)
            res = resolve(paths, fields, cfg, auto_threshold=cfg["merge_modes"]["balanced"]["entity_merge"], records=records)
            seconds = time.perf_counter() - t0
        groups = defaultdict(list)
        for rec, ent in res.assignment.items():
            groups[ent].append(rec)
        m = prf(cluster_pairs(groups.values()), truth_pairs({k: gt["entities"][v] for k, v in ids.items()}, list(ids)))
        row = {"entities": n, "records": len(records), "candidate_pairs": res.stats["candidate_pairs"], "seconds": seconds,
               "peak_rss_mb": mem.peak_mb, "records_per_second": len(records) / seconds, "f1": m["f1"],
               "blocking_ms": res.stats["timings_ms"]["blocking"], "scoring_ms": res.stats["timings_ms"]["scoring"], "clustering_ms": res.stats["timings_ms"]["clustering"]}
        rows.append(row)
        print(f"  {n:6d} entities: {row['records']:6d} records, {row['candidate_pairs']:8d} pairs, {seconds:6.1f}s, {mem.peak_mb:6.0f} MB, F1 {m['f1']:.3f}")
    save_json(f"er_scaling{suffix}.json", {"refine_block_size": cfg["entity_resolution"].get("refine_block_size"), "rows": rows})
    md = [f"# Entity resolution scaling (single process, adaptive block refinement {'off' if args.no_refine else 'on'})", "",
          markdown_table(rows, ["entities", "records", "candidate_pairs", "seconds", "records_per_second", "peak_rss_mb", "f1", "blocking_ms", "scoring_ms", "clustering_ms"],
                         ["entities", "records", "candidate pairs", "seconds", "records/s", "peak RSS (MB)", "pairwise F1", "blocking (ms)", "scoring (ms)", "clustering (ms)"],
                         {"seconds": ".1f", "records_per_second": ".0f", "peak_rss_mb": ".0f", "f1": ".3f", "blocking_ms": ".0f", "scoring_ms": ".0f", "clustering_ms": ".0f"})]
    (RESULTS / f"er_scaling{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
