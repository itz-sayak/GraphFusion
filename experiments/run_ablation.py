"""Schema-matching ablation: does graph reasoning improve integration quality?

Methods (all run through the same matcher code path):

    exact     normalised exact column-name equality (baseline)
    A         name similarity only
    B         name + datatype
    C         name + datatype + instance signals (values, distribution, pattern, cardinality)
    D         C + semantic similarity (thesaurus + embeddings)
    E1        D + bipartite 1:1 alignment
    E2        D + similarity flooding + bipartite alignment   (full graph-based system)
    F         learned calibrated classifier over the same evidence + flooding features + bipartite alignment
    G         blend: mean of the weighted score (E2) and the learned probability (F)

Evaluation sets: the fabricated Valentine-style benchmark (easy / medium / hard),
the sample customer scenario and the real NYC mobility scenario.

Two numbers per method, because the score scale changes when signals are
removed: F1 at the system's default acceptance threshold (0.55) and the best
F1 over all thresholds (ranking quality, independent of calibration).

    python experiments/run_ablation.py [--per-level 8]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from common import RESULTS, ROOT, grouped_bars, markdown_table, mean, prf, save_json

from backend.config import load_config
from backend.ingestion.loader import ingest_file
from backend.matching.aligner import align
from backend.matching.matcher import MatchStrategy, match_schemas
from backend.profiling.profiler import profile_dataset
from backend.storage.workspace import Workspace
from fabricator import make_benchmark

INSTANCE = ("value_overlap", "distribution", "pattern", "cardinality")
METHODS = {
    "A": MatchStrategy(("name",), flooding=False, bipartite=False, name="A"),
    "B": MatchStrategy(("name", "datatype"), flooding=False, bipartite=False, name="B"),
    "C": MatchStrategy(("name", "datatype", *INSTANCE), flooding=False, bipartite=False, name="C"),
    "D": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=False, bipartite=False, name="D"),
    "E1": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=False, bipartite=True, name="E1"),
    "E2": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=True, bipartite=True, name="E2"),
    # learned scorers (model trained on fabricated scenarios with a disjoint seed: experiments/train_matcher.py)
    "F": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=True, bipartite=True, name="F", scorer="learned"),
    "G": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=True, bipartite=True, name="G", scorer="blend"),
    "H": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=True, bipartite=True, name="H"),
    "I": MatchStrategy(("name", "datatype", "semantic", *INSTANCE), flooding=True, bipartite=True, name="I"),
}
LABELS = {"exact": "Exact name", "A": "A name", "B": "B +type", "C": "C +values", "D": "D +semantic", "E1": "E1 +bipartite", "E2": "E2 full graph", "F": "F learned", "G": "G blend", "H": "H MiniLM encoder", "I": "I bge-small encoder"}
THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(27)]
DEFAULT_T = 0.55
# at most 8 series (fixed categorical palette); A and B are in the tables
CHART_METHODS = ["exact", "D", "E2", "F", "G", "H", "I"]
# encoder test: E2 with a pretrained sentence encoder instead of character TF-IDF for the embedding part of S_E
ENCODERS = {"H": "sentence-transformers/all-MiniLM-L6-v2", "I": "BAAI/bge-small-en-v1.5"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def profile_files(files: dict[str, Path], ws: Workspace, cfg: dict):
    profiles, sketches = {}, {}
    for ds, path in files.items():
        art = ingest_file(ws, path, source_name=f"{ds}{path.suffix}")
        p, s = profile_dataset(art, cfg)
        profiles[art.dataset_id], sketches[art.dataset_id] = p, s
    return profiles, sketches


def evaluate(profiles, sketches, truth: set, cfg: dict) -> dict[str, dict]:
    out = {}
    exact = set()
    ds = list(profiles)
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            for ca in profiles[a].columns:
                for cb in profiles[b].columns:
                    if _norm(ca.name) == _norm(cb.name):
                        exact.add(tuple(sorted((f"{a}::{ca.name}", f"{b}::{cb.name}"))))
    m = prf(exact, truth)
    out["exact"] = {"default": m, "best": m, "seconds": 0.0}
    uniq = {f"{d}::{c.name}": c.uniqueness for d, p in profiles.items() for c in p.columns}
    for key, strategy in METHODS.items():
        mcfg = cfg
        if key in ENCODERS:
            mcfg = copy.deepcopy(cfg)
            mcfg["embeddings"] = {"backend": "sentence_transformers", "model": ENCODERS[key]}
            from backend.matching.embeddings import make_encoder

            assert make_encoder(mcfg).name == "sentence_transformers", "encoder test requires sentence-transformers"
        t0 = time.perf_counter()
        res = match_schemas(profiles, sketches, mcfg, strategy, min_accept=DEFAULT_T)
        elapsed = time.perf_counter() - t0
        per_t = {}
        for t in THRESHOLDS:
            align(res.matches, t, cfg["merge"]["key_uniqueness"], cfg["schema_matching"]["alignment"]["fk_min_containment"], uniq,
                  allow_fk=strategy.bipartite, one_to_one=strategy.bipartite)
            per_t[t] = prf({tuple(sorted((m.left.key, m.right.key))) for m in res.matches if m.accepted}, truth)
        best_t = max(per_t, key=lambda t: (per_t[t]["f1"], -abs(t - DEFAULT_T)))
        out[key] = {"default": per_t[DEFAULT_T], "best": {**per_t[best_t], "threshold": best_t}, "seconds": round(elapsed, 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-level", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_ablation_"))
    rows: list[dict] = []
    try:
        scenarios = make_benchmark(tmp / "bench", per_level=args.per_level)
        for sc in scenarios:
            ws = Workspace(tmp / "ws", sc.scenario_id)
            profiles, sketches = profile_files(sc.files, ws, cfg)
            res = evaluate(profiles, sketches, sc.truth, cfg)
            for method, r in res.items():
                rows.append({"set": f"fabricated-{sc.difficulty}", "scenario": sc.scenario_id, "domain": sc.domain, "method": method, **{f"{k}_default": v for k, v in r["default"].items()}, **{f"{k}_best": v for k, v in r["best"].items()}, "seconds": r["seconds"]})
            print(f"  {sc.scenario_id} ({sc.domain}, {sc.difficulty}): " + "  ".join(f"{m}={r['best']['f1']:.2f}" for m, r in res.items()))

        real_sets = [("sample", {"sample_customers": ROOT / "data/sample/sample_customers.csv", "sample_customer_master": ROOT / "data/sample/sample_customer_master.json", "sample_sales": ROOT / "data/sample/sample_sales.parquet"},
                      ROOT / "experiments/ground_truth/sample_ground_truth.json")]
        nyc = ROOT / "data/processed/nyc"
        if (nyc / "yellow_tripdata_sample.parquet").exists():
            real_sets.append(("nyc", {"yellow_tripdata_sample": nyc / "yellow_tripdata_sample.parquet", "taxi_zone_lookup": nyc / "taxi_zone_lookup.csv", "nta_demographics": nyc / "nta_demographics.json", "census_acs_nyc_counties": nyc / "census_acs_nyc_counties.json"},
                              ROOT / "experiments/ground_truth/nyc_mappings.json"))
        for name, files, gt_path in real_sets:
            truth = {tuple(sorted(p)) for p in json.loads(gt_path.read_text(encoding="utf-8"))["column_matches"]}
            profiles, sketches = profile_files(files, Workspace(tmp / "ws", name), cfg)
            res = evaluate(profiles, sketches, truth, cfg)
            for method, r in res.items():
                rows.append({"set": name, "scenario": name, "domain": name, "method": method, **{f"{k}_default": v for k, v in r["default"].items()}, **{f"{k}_best": v for k, v in r["best"].items()}, "seconds": r["seconds"]})
            print(f"  {name}: " + "  ".join(f"{m}={r['default']['f1']:.2f}/{r['best']['f1']:.2f}" for m, r in res.items()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # micro-averaged over each evaluation set (sum TP / predicted / truth), at default threshold; macro best-F1
    summary = []
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["method"])].append(r)
    sets = ["fabricated-easy", "fabricated-medium", "fabricated-hard", "sample", "nyc"]
    for s in sets:
        for method in ["exact", *METHODS]:
            rs = by.get((s, method))
            if not rs:
                continue
            tp, pred, tru = sum(r["tp_default"] for r in rs), sum(r["predicted_default"] for r in rs), sum(r["truth_default"] for r in rs)
            p = tp / pred if pred else 0.0
            rc = tp / tru if tru else 0.0
            summary.append({"set": s, "method": method, "label": LABELS[method], "precision": round(p, 3), "recall": round(rc, 3),
                            "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0, "best_f1": round(mean(r["f1_best"] for r in rs), 3), "scenarios": len(rs),
                            "seconds": round(mean(r["seconds"] for r in rs), 3)})
    save_json("ablation_raw.json", rows)
    save_json("ablation_summary.json", summary)

    md = ["# Schema-matching ablation", "", f"Default acceptance threshold {DEFAULT_T}; P/R/F1 micro-averaged per set; best F1 = mean over scenarios of the best threshold.", ""]
    for s in sets:
        part = [r for r in summary if r["set"] == s]
        if part:
            md += [f"## {s} ({part[0]['scenarios']} scenario{'s' if part[0]['scenarios'] > 1 else ''})", "",
                   markdown_table(part, ["label", "precision", "recall", "f1", "best_f1", "seconds"], ["method", "precision", "recall", "F1 @0.55", "best F1", "match time (s)"], {"precision": ".3f", "recall": ".3f", "f1": ".3f", "best_f1": ".3f", "seconds": ".3f"}), ""]
    (RESULTS / "ablation.md").write_text("\n".join(md), encoding="utf-8")

    present = [s for s in sets if any(r["set"] == s for r in summary)]
    for metric, fname, title in (("f1", "ablation_f1_default.png", "Schema matching F1 at the default threshold (0.55)"), ("best_f1", "ablation_f1_best.png", "Schema matching best F1 over thresholds")):
        grouped_bars(RESULTS / fname, present, {LABELS[m]: [next((r[metric] for r in summary if r["set"] == s and r["method"] == m), 0.0) for s in present] for m in CHART_METHODS}, title + " (table has all methods)", "F1", ylim=(0, 1.12))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
