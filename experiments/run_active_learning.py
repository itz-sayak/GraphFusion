"""Active learning for entity resolution: how much do a few human labels help?

Harder variant of the sample scenario: 60% of e-mails and phone numbers are
missing in both customer systems, so linkage must rely on names and cities
and many pairs land in the uncertain band.

A simulated oracle (the hidden ground truth) labels record pairs in batches
of 10. After each batch, entity resolution is re-run with the labels applied
(labelled pairs become must-/cannot-link constraints in correlation
clustering). Three ways of choosing which pairs to label:

    uncertainty   margin sampling around the balanced record-link threshold  — backend/entity_resolution/resolver.py::review_queue
    random        uniformly random among compared pairs
    hybrid        70% uncertainty + 30% random per batch (the app's review queue)

and four ways of using the labels (constraints only; + m update; + calibration on
all labels; + calibration on the randomly sampled labels only).

Pairwise F1 against ground truth is reported at 0, 10, 25, 50 and 100 labels,
averaged over generator seeds.

    python experiments/run_active_learning.py [--seeds 7 11 23]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from common import RESULTS, line_chart, markdown_table, mean, prf, save_json
from run_evaluation import cluster_pairs, truth_pairs

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve, review_queue
from backend.ingestion.loader import ingest_file
from backend.storage.workspace import Workspace

BUDGETS = [0, 10, 25, 50, 100]
VARIANTS = [("uncertainty", "constraints"), ("random", "constraints"), ("uncertainty", "constraints + m update"),
            ("uncertainty", "constraints + calibration"), ("random", "constraints + calibration"), ("hybrid", "constraints + random-only calibration")]
VARIANT_CONFIG = {
    "constraints": {"label_weight": 0.0, "label_calibration": "off"},
    "constraints + m update": {"label_weight": 5.0, "label_calibration": "off"},
    "constraints + calibration": {"label_weight": 0.0, "label_calibration": "all"},
    "constraints + random-only calibration": {"label_weight": 0.0, "label_calibration": "random_only"},
}
RANDOM_SHARE = 0.3  # hybrid: share of each batch drawn at random (the default review queue in the app)
BATCH = 10
ENTITY_T = load_config()["merge_modes"]["balanced"]["entity_merge"]
HARD = {"crm_email_missing": 0.6, "crm_phone_missing": 0.6, "mdm_email_missing": 0.6, "mdm_phone_missing": 0.6}


def f1_of(res, truth) -> dict:
    groups = defaultdict(list)
    for rec, ent in res.assignment.items():
        groups[ent].append(rec)
    return prf(cluster_pairs(groups.values()), truth)


def run_seed(seed: int, tmp: Path, cfg: dict) -> list[dict]:
    d = tmp / f"seed{seed}"
    make_sample_data.generate(d, seed=seed, **HARD)
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    ws = Workspace(tmp / "ws", f"al{seed}")
    a = ingest_file(ws, d / "sample_customers.csv")
    b = ingest_file(ws, d / "sample_customer_master.json")
    ids = {a.dataset_id: "sample_customers", b.dataset_id: "sample_customer_master"}
    entity_of = {(ds, row): ent for ds, key in ids.items() for row, ent in enumerate(gt["entities"][key])}
    fields = [
        FieldSpec("name", S.NAME, {a.dataset_id: "customer_name", b.dataset_id: "full_name"}),
        FieldSpec("city", S.CITY, {a.dataset_id: "city", b.dataset_id: "location"}, "semantic:CITY"),
        FieldSpec("email", S.EMAIL, {a.dataset_id: "email", b.dataset_id: "email_address"}),
        FieldSpec("phone", S.PHONE, {a.dataset_id: "phone", b.dataset_id: "phone_number"}, "digits"),
    ]
    paths = {a.dataset_id: a.storage_path, b.dataset_id: b.storage_path}
    records = load_records(paths, fields)
    truth = truth_pairs({k: gt["entities"][v] for k, v in ids.items()}, list(ids))

    def oracle(left, right) -> bool:
        return entity_of[left] == entity_of[right]

    rows = []
    for strategy, variant in VARIANTS:
        vcfg = copy.deepcopy(cfg)
        vcfg["entity_resolution"].update(VARIANT_CONFIG[variant])
        rng = random.Random(seed)
        labels: dict = {}
        random_pairs: set = set()
        res = resolve(paths, fields, vcfg, auto_threshold=ENTITY_T, records=records)
        for budget in BUDGETS:
            while len(labels) < budget:
                need = min(BATCH, budget - len(labels))
                n_rand = need if strategy == "random" else (round(need * RANDOM_SHARE) if strategy == "hybrid" else 0)
                picks = [((q["left"]["dataset"], q["left"]["row"]), (q["right"]["dataset"], q["right"]["row"]))
                         for q in review_queue(res, limit=need - n_rand, threshold=ENTITY_T, exclude=set(labels))] if need - n_rand > 0 else []
                pool = sorted(p for p in res.pair_probabilities if p not in labels and p not in picks)
                rand = rng.sample(pool, min(n_rand, len(pool)))
                random_pairs.update(rand)
                picks += rand
                if not picks:
                    break
                for left, right in picks:
                    labels[(left, right)] = oracle(left, right)
                res = resolve(paths, fields, vcfg, auto_threshold=ENTITY_T, records=records, labels=labels, calibration_pairs=random_pairs)
            m = f1_of(res, truth)
            rows.append({"seed": seed, "strategy": strategy, "variant": variant, "labels": len(labels), "budget": budget, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                         "labelled_matches": sum(labels.values()), "uncertain_pairs_left": sum(1 for p in res.pair_probabilities.values() if 0.05 < p < 0.95)})
            print(f"  seed {seed} {strategy:11s} {variant:24s} labels {len(labels):3d}: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}  (labelled matches {sum(labels.values())})")
    return rows


def plot(summary: list[dict]) -> None:
    lows = min(r["f1"] for r in summary)
    line_chart(RESULTS / "active_learning.png", list(range(len(BUDGETS))),
               {f"{s} · {v}": [r["f1"] for r in summary if r["strategy"] == s and r["variant"] == v] for s, v in VARIANTS},
               "Entity resolution F1 vs number of human labels", "labels", "pairwise F1 (mean of seeds)", value_fmt="{:.3f}",
               xticklabels=[str(b) for b in BUDGETS], ylim=(max(0.0, round(lows - 0.05, 1)), 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--replot", action="store_true", help="redraw the chart from results/active_learning.json")
    args = ap.parse_args()
    if args.replot:
        plot(json.loads((RESULTS / "active_learning.json").read_text(encoding="utf-8"))["summary"])
        return
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_al_"))
    rows: list[dict] = []
    try:
        for seed in args.seeds:
            rows += run_seed(seed, tmp, cfg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    summary = []
    for strategy, variant in VARIANTS:
        for budget in BUDGETS:
            rs = [r for r in rows if r["strategy"] == strategy and r["variant"] == variant and r["budget"] == budget]
            summary.append({"strategy": strategy, "variant": variant, "labels": budget, **{k: round(mean(r[k] for r in rs), 4) for k in ("precision", "recall", "f1", "labelled_matches", "uncertain_pairs_left")}})
    save_json("active_learning.json", {"seeds": args.seeds, "hard_variant": HARD, "rows": rows, "summary": summary})
    md = ["# Active learning for entity resolution", "",
          f"Hard variant of the sample scenario ({', '.join(f'{k}={v}' for k, v in HARD.items())}); seeds {args.seeds}; oracle = ground truth; labels in batches of {BATCH}; record-link auto-merge threshold {ENTITY_T} (balanced mode, calibrated probabilities).", "",
          "Variants: **constraints** = labels become must-/cannot-link constraints only; **+ m update** = labelled matches also add evidence to the Fellegi–Sunter m estimates; **+ calibration** = semi-supervised Platt recalibration of match weights using all labels; **+ random-only calibration** (app default) = recalibration uses only the randomly sampled labels. **hybrid** = each batch of 10 is 7 uncertainty-sampled + 3 random pairs.", "",
          markdown_table(summary, ["strategy", "variant", "labels", "precision", "recall", "f1", "labelled_matches", "uncertain_pairs_left"],
                         ["selection", "label use", "labels", "precision", "recall", "pairwise F1", "labelled pairs that were matches", "uncertain pairs left"],
                         {"precision": ".3f", "recall": ".3f", "f1": ".3f", "labelled_matches": ".1f", "uncertain_pairs_left": ".0f"}), "",
          "![F1 vs labels](active_learning.png)"]
    (RESULTS / "active_learning.md").write_text("\n".join(md), encoding="utf-8")
    plot(summary)
    print("\n".join(md))


if __name__ == "__main__":
    main()
