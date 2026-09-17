"""Is the remaining mid-band under-confidence caused by field dependence?

Log-linear Fellegi–Sunter with interaction terms helps only if comparison fields are dependent among
matches (e.g. a true match that agrees on name also tends to agree on city beyond what the marginals
predict). For each field pair this measures, among TRUE matches (oracle) and posterior-weighted among
candidates, the lift  P(exact, exact) / (P(exact) · P(exact)); lift ≈ 1 means independence.
It also lists the comparison patterns behind the pairs predicted 0.5–0.9, with predicted vs observed
match rate. Hard customer scenario (60% missing e-mail/phone), seeds 7/11/23, calibrated prior.

    python experiments/run_field_dependence.py
"""
from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

from common import RESULTS, markdown_table, save_json

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.comparators import comparator_for
from backend.entity_resolution.fellegi_sunter import compare_pair
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve
from backend.ingestion.loader import ingest_file
from backend.storage.workspace import Workspace

HARD = {"crm_email_missing": 0.6, "crm_phone_missing": 0.6, "mdm_email_missing": 0.6, "mdm_phone_missing": 0.6}
PAIRS = [("name", "city"), ("name", "email"), ("name", "phone"), ("city", "email"), ("city", "phone"), ("email", "phone")]


def main() -> None:
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_dep_"))
    rows_all: list[tuple[dict, float, int]] = []
    for seed in (7, 11, 23):
        d = tmp / f"h{seed}"
        make_sample_data.generate(d, seed=seed, **HARD)
        gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
        ws = Workspace(tmp / "ws", f"d{seed}")
        a = ingest_file(ws, d / "sample_customers.csv")
        b = ingest_file(ws, d / "sample_customer_master.json")
        ids = {a.dataset_id: "sample_customers", b.dataset_id: "sample_customer_master"}
        ent = {(ds, r): e for ds, k in ids.items() for r, e in enumerate(gt["entities"][k])}
        fields = [
            FieldSpec("name", S.NAME, {a.dataset_id: "customer_name", b.dataset_id: "full_name"}),
            FieldSpec("city", S.CITY, {a.dataset_id: "city", b.dataset_id: "location"}, "semantic:CITY"),
            FieldSpec("email", S.EMAIL, {a.dataset_id: "email", b.dataset_id: "email_address"}),
            FieldSpec("phone", S.PHONE, {a.dataset_id: "phone", b.dataset_id: "phone_number"}, "digits"),
        ]
        paths = {a.dataset_id: a.storage_path, b.dataset_id: b.storage_path}
        records = load_records(paths, fields)
        res = resolve(paths, fields, cfg, auto_threshold=0.5, records=records)
        comps = {f.name: comparator_for(f.semantic_type, f.normalizer)[1] for f in fields}
        index = {ds: i for i, ds in enumerate(paths)}
        for (x, y), p in res.pair_probabilities.items():
            levels, _ = compare_pair(records[(index[x[0]], x[1])], records[(index[y[0]], y[1])], comps)
            rows_all.append((levels, p, int(ent[x] == ent[y])))

    lift_rows = []
    for f1, f2 in PAIRS:
        for basis, weight in (("true matches", lambda p, y: y), ("posterior-weighted", lambda p, y: p)):
            present = [(lv, weight(p, y)) for lv, p, y in rows_all if lv[f1] != "null" and lv[f2] != "null"]
            tot = sum(w for _, w in present)
            if tot < 20:
                continue
            e1 = sum(w for lv, w in present if lv[f1] == "exact") / tot
            e2 = sum(w for lv, w in present if lv[f2] == "exact") / tot
            ee = sum(w for lv, w in present if lv[f1] == "exact" and lv[f2] == "exact") / tot
            lift_rows.append({"fields": f"{f1} × {f2}", "basis": basis, "weight": tot, "p1": e1, "p2": e2, "joint": ee, "lift": ee / (e1 * e2) if e1 * e2 else None})

    patterns: dict[tuple, list] = defaultdict(lambda: [0, 0, 0.0])
    for lv, p, y in rows_all:
        if 0.5 <= p < 0.9:
            k = tuple(lv[f] for f in ("name", "city", "email", "phone"))
            patterns[k][0] += 1
            patterns[k][1] += y
            patterns[k][2] += p
    pattern_rows = [{"pattern": " / ".join(k), "pairs": n, "predicted": pp / n, "observed": yy / n}
                    for k, (n, yy, pp) in sorted(patterns.items(), key=lambda kv: -kv[1][0]) if n >= 5]

    save_json("field_dependence.json", {"lift": lift_rows, "mid_band_patterns": pattern_rows})
    md = ["# Field dependence among matches (hard customer scenario, 3 seeds)", "",
          "Lift = P(both fields exact) / (P(first exact) · P(second exact)). Lift ≈ 1 means the fields agree independently, so interaction terms cannot change the posterior.", "",
          markdown_table(lift_rows, ["fields", "basis", "weight", "p1", "p2", "joint", "lift"],
                         ["field pair", "basis", "pairs (weight)", "P(first exact)", "P(second exact)", "P(both exact)", "lift"],
                         {"weight": ".0f", "p1": ".3f", "p2": ".3f", "joint": ".3f", "lift": ".3f"}), "",
          "## Comparison patterns of pairs predicted 0.5–0.9 (name / city / email / phone)", "",
          markdown_table(pattern_rows, ["pattern", "pairs", "predicted", "observed"], ["levels", "pairs", "mean predicted", "observed match rate"],
                         {"predicted": ".3f", "observed": ".3f"})]
    (RESULTS / "field_dependence.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
