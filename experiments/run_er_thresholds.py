"""Record-link thresholds under the old and the calibrated prior, on tuning and held-out seeds.

The merge-mode record-link thresholds (strict 0.90 / balanced 0.50 / permissive 0.35) were chosen by
principle on seeds 7/11/23 after calibrating the prior; seeds 101/202/303 were not looked at before.
Pairwise precision / recall / F1 of correlation clustering for the default and the hard (sparse)
customer scenarios.

    python experiments/run_er_thresholds.py
"""
from __future__ import annotations

import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from common import RESULTS, markdown_table, mean, prf, save_json
from run_evaluation import cluster_pairs, truth_pairs

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve
from backend.ingestion.loader import ingest_file
from backend.storage.workspace import Workspace

SCENARIOS = {"default": {}, "hard": {"crm_email_missing": 0.6, "crm_phone_missing": 0.6, "mdm_email_missing": 0.6, "mdm_phone_missing": 0.6}}
SEED_SETS = {"tuning (7, 11, 23)": (7, 11, 23), "held-out (101, 202, 303)": (101, 202, 303)}
SETTINGS = [
    ("old prior, old balanced", "candidates", 0.90),
    ("calibrated, strict", "all_pairs", 0.90),
    ("calibrated, balanced", "all_pairs", 0.50),
    ("calibrated, permissive", "all_pairs", 0.35),
]


def main() -> None:
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_thr_"))
    acc: dict[tuple, list] = defaultdict(list)
    for scenario, kw in SCENARIOS.items():
        for seed_set, seeds in SEED_SETS.items():
            for seed in seeds:
                d = tmp / f"{scenario}{seed}"
                make_sample_data.generate(d, seed=seed, **kw)
                gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
                ws = Workspace(tmp / "ws", f"t{scenario}{seed}")
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
                records = load_records(paths, fields)
                truth = truth_pairs({k: gt["entities"][v] for k, v in ids.items()}, list(ids))
                for name, population, threshold in SETTINGS:
                    c = copy.deepcopy(cfg)
                    c["entity_resolution"]["prior_population"] = population
                    res = resolve(paths, fields, c, auto_threshold=threshold, records=records)
                    groups = defaultdict(list)
                    for rec, ent in res.assignment.items():
                        groups[ent].append(rec)
                    acc[(scenario, seed_set, name)].append(prf(cluster_pairs(groups.values()), truth))
                print(f"  {scenario} seed {seed} done")
    rows = [{"scenario": sc, "seeds": ss, "setting": name, "threshold": thr,
             "precision": mean(m["precision"] for m in acc[(sc, ss, name)]), "recall": mean(m["recall"] for m in acc[(sc, ss, name)]),
             "f1": mean(m["f1"] for m in acc[(sc, ss, name)])}
            for sc in SCENARIOS for ss in SEED_SETS for name, _, thr in SETTINGS]
    save_json("er_thresholds.json", {"rows": rows})
    md = ["# Record-link thresholds: old vs calibrated prior", "",
          "Pairwise P/R/F1 (correlation clustering), mean of 3 seeds. Thresholds were chosen on the tuning seeds; the held-out seeds were not used for any choice.", "",
          markdown_table(rows, ["scenario", "seeds", "setting", "threshold", "precision", "recall", "f1"],
                         ["scenario", "seeds", "setting", "record-link threshold", "precision", "recall", "pairwise F1"],
                         {"threshold": ".2f", "precision": ".3f", "recall": ".3f", "f1": ".3f"})]
    (RESULTS / "er_thresholds.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
