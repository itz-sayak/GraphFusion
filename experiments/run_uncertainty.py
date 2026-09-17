"""Are the per-row match probabilities calibrated?

``_match_probability`` of a fused-entity row is its cluster confidence (mean
Fellegi–Sunter probability of the compared pairs inside the cluster). If it is
calibrated, rows with p ≈ 0.8 should be correct ~80% of the time, and the
uncertainty-aware aggregates (expected value Σp, variance Σp(1−p)) are honest.

Correctness of a multi-record entity = purity: all its records belong to one
true entity (singletons are trivially pure and excluded). Scenarios: the default
sample generator and the hard variant (60% missing e-mail/phone), seeds 7/11/23.
Variants:

    uncalibrated      Fellegi–Sunter probabilities as estimated by EM
    + 30 random labels  oracle labels on 30 uniformly sampled compared pairs, used as
                      constraints and for random-only label calibration (resolver)

Metrics: expected calibration error (10 equal-mass bins), Brier score, mean predicted vs observed purity, and whether
the observed number of pure entities lies inside the 95% interval Σp ± 1.96·√Σp(1−p)
that ``aggregate_with_uncertainty`` would report for a count.

    python experiments/run_uncertainty.py [--seeds 7 11 23] [--replot]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

from common import RESULTS, grouped_bars, markdown_table, mean, prf, save_json
from run_evaluation import cluster_pairs, truth_pairs

import make_sample_data
from backend.config import load_config
from backend.core.models import SemanticType as S
from backend.entity_resolution.resolver import FieldSpec, load_records, resolve
from backend.ingestion.loader import ingest_file
from backend.storage.workspace import Workspace

SCENARIOS = {"default": {}, "hard": {"crm_email_missing": 0.6, "crm_phone_missing": 0.6, "mdm_email_missing": 0.6, "mdm_phone_missing": 0.6}}
N_LABELS = 30
# (name, entity_resolution config overrides, with 30 random labels)
# (name, entity_resolution overrides, with random labels, auto-merge threshold used for the merged-entity metrics)
# the old prior was used with the old balanced threshold 0.9; calibrated probabilities with the new balanced 0.5
VARIANTS = [
    ("old prior (t=0.9)", {"prior_population": "candidates", "term_frequency": False}, False, 0.9),
    ("old prior + TF (t=0.9)", {"prior_population": "candidates", "term_frequency": True}, False, 0.9),
    ("calibrated (t=0.5)", {"prior_population": "all_pairs", "term_frequency": False}, False, 0.5),
    ("calibrated + TF (t=0.5)", {"prior_population": "all_pairs", "term_frequency": True}, False, 0.5),
    (f"calibrated + {N_LABELS} labels (t=0.5)", {"prior_population": "all_pairs", "term_frequency": False}, True, 0.5),
]
BINS = 10


def ece(ps: list[float], ys: list[int], bins: int = BINS) -> tuple[float, list[dict]]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    size = max(1, math.ceil(len(order) / bins))
    table, err = [], 0.0
    for b in range(0, len(order), size):
        idx = order[b:b + size]
        conf = mean(ps[i] for i in idx)
        acc = mean(ys[i] for i in idx)
        err += len(idx) / len(ps) * abs(conf - acc)
        table.append({"predicted": conf, "observed": acc, "n": len(idx)})
    return err, table


def pair_calibration_error(ps: list[float], ys: list[int], bins: int = BINS) -> float:
    """ECE over compared pairs with 10 equal-width probability bins (non-matches near 0 no longer dominate one bin)."""
    total = len(ps)
    if not total:
        return 0.0
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(ps) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            err += len(idx) / total * abs(mean(ps[i] for i in idx) - mean(ys[i] for i in idx))
    return err


def run_seed(scenario: str, seed: int, tmp: Path, cfg: dict) -> list[dict]:
    d = tmp / f"{scenario}{seed}"
    make_sample_data.generate(d, seed=seed, **SCENARIOS[scenario])
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    ws = Workspace(tmp / "ws", f"u{scenario}{seed}")
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
    out = []
    truth = truth_pairs({k: gt["entities"][v] for k, v in ids.items()}, list(ids))
    rng = random.Random(seed)
    for variant, overrides, with_labels, t_merge in VARIANTS:
        vcfg = copy.deepcopy(cfg)
        vcfg["entity_resolution"].update(overrides)
        res = resolve(paths, fields, vcfg, auto_threshold=t_merge, records=records)
        if with_labels:
            sample = rng.sample(sorted(res.pair_probabilities), min(N_LABELS, len(res.pair_probabilities)))
            labels = {pr: entity_of[pr[0]] == entity_of[pr[1]] for pr in sample}
            res = resolve(paths, fields, vcfg, auto_threshold=t_merge, records=records, labels=labels, calibration_pairs=set(sample))
        res09 = res if t_merge == 0.9 else resolve(paths, fields, vcfg, auto_threshold=0.9, records=records)
        res05 = res if t_merge == 0.5 else resolve(paths, fields, vcfg, auto_threshold=0.5, records=records)
        ps, ys = [], []
        for c in res.clusters:
            if c.size < 2:
                continue
            ps.append(c.confidence)
            ys.append(int(len({entity_of[m] for m in c.members}) == 1))
        f1s = {}
        for t, r in ((0.9, res09), (0.5, res05)):
            groups = defaultdict(list)
            for rec, ent in r.assignment.items():
                groups[ent].append(rec)
            f1s[f"f1_{t}"] = prf(cluster_pairs(groups.values()), truth)["f1"]
        # pair level: every compared pair's probability against the truth (independent of any merge threshold)
        pp = list(res.pair_probabilities.items())
        pair_p = [pr for _, pr in pp]
        pair_y = [int(entity_of[a] == entity_of[b]) for (a, b), _ in pp]
        review = sum(1 for pr in pair_p if 0.7 <= pr < 0.9)
        out.append({"scenario": scenario, "seed": seed, "variant": variant, "p": ps, "y": ys, "prior": res.stats.get("prior"), **f1s,
                    "pair_p": pair_p, "pair_y": pair_y, "review_band": review})
        print(f"  {scenario:8s} seed {seed} {variant:26s} entities {len(ps)}  mean p {mean(ps):.3f}  purity {mean(ys):.3f}  F1@0.9 {f1s['f1_0.9']:.3f}  F1@0.5 {f1s['f1_0.5']:.3f}")
    return out


def summarise(runs: list[dict]) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for variant in dict.fromkeys(r["variant"] for r in runs):
            rs = [r for r in runs if r["scenario"] == scenario and r["variant"] == variant]
            ps = [p for r in rs for p in r["p"]]
            ys = [y for r in rs for y in r["y"]]
            e, _ = ece(ps, ys)
            covered = 0
            for r in rs:
                exp = sum(r["p"])
                sd = math.sqrt(sum(p * (1 - p) for p in r["p"]))
                covered += int(abs(sum(r["y"]) - exp) <= 1.96 * sd)
            qp = [p for r in rs for p in r["pair_p"]]
            qy = [y for r in rs for y in r["pair_y"]]
            pair_ece = pair_calibration_error(qp, qy)
            eps = 1e-6
            logloss = -mean(y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p)) for p, y in zip(qp, qy))
            rows.append({"scenario": scenario, "variant": variant, "f1_09": mean(r["f1_0.9"] for r in rs), "f1_05": mean(r["f1_0.5"] for r in rs),
                         "pair_ece": pair_ece, "pair_brier": mean((p - y) ** 2 for p, y in zip(qp, qy)), "pair_logloss": logloss,
                         "review_band": mean(r["review_band"] for r in rs), "entities": len(ps), "mean_p": mean(ps), "purity": mean(ys), "ece": e,
                         "brier": mean((p - y) ** 2 for p, y in zip(ps, ys)), "expected_pure": mean(sum(r["p"]) for r in rs),
                         "observed_pure": mean(sum(r["y"]) for r in rs), "ci_covered": f"{covered}/{len(rs)}"})
    return rows


def plot(runs: list[dict]) -> None:
    """Pair-level reliability diagram: mean predicted probability vs observed match rate per probability band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import GRID, INK, INK_2, SERIES, SURFACE, _style

    bands = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.0001)]
    shown = [("old prior (t=0.9)", "old prior (blocked candidates)", SERIES[1]), ("calibrated (t=0.5)", "prior over all pairs (default)", SERIES[0])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=150, sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        _style(ax, f"{scenario} scenario", "observed match rate" if scenario == "default" else None)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", zorder=1)
        ax.annotate("perfect calibration", (0.62, 0.55), color=INK_2, fontsize=8, rotation=36)
        for variant, label, color in shown:
            rs = [r for r in runs if r["scenario"] == scenario and r["variant"] == variant]
            ps = [p for r in rs for p in r["pair_p"]]
            ys = [y for r in rs for y in r["pair_y"]]
            xs, obs = [], []
            for lo, hi in bands:
                idx = [i for i, p in enumerate(ps) if lo <= p < hi]
                if len(idx) >= 20:  # bands with too few pairs are noise
                    xs.append(mean(ps[i] for i in idx))
                    obs.append(mean(ys[i] for i in idx))
            ax.plot(xs, obs, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("predicted P(same entity), mean per band", color=INK_2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.suptitle("Record-link probabilities: predicted vs observed (all compared pairs, 3 seeds)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(RESULTS / "uncertainty_calibration.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.replot:
        plot(json.loads((RESULTS / "uncertainty.json").read_text(encoding="utf-8"))["runs"])
        return
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_unc_"))
    runs: list[dict] = []
    try:
        for scenario in SCENARIOS:
            for seed in args.seeds:
                runs += run_seed(scenario, seed, tmp, cfg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    summary = summarise(runs)
    save_json("uncertainty.json", {"seeds": args.seeds, "scenarios": SCENARIOS, "summary": summary, "runs": runs})
    md = ["# Calibration of match probabilities (probabilistic provenance)", "",
          f"Multi-record entities from the customer scenarios, seeds {args.seeds}. Predicted = cluster confidence (the `_match_probability` of the entity's output row); "
          "observed = purity (all records belong to one true entity). ECE uses 10 equal-mass bins. "
          "\"95% interval covers\" counts the seeds where the observed number of pure entities lies in Σp ± 1.96·√Σp(1−p).", "",
          markdown_table(summary, ["scenario", "variant", "entities", "mean_p", "purity", "ece", "brier", "expected_pure", "observed_pure", "ci_covered", "f1_09", "f1_05"],
                         ["scenario", "probabilities", "entities (pooled)", "mean predicted", "observed purity", "ECE", "Brier", "expected pure / seed", "observed pure / seed", "95% interval covers", "pairwise F1 @0.9", "pairwise F1 @0.5"],
                         {"mean_p": ".3f", "purity": ".3f", "ece": ".3f", "brier": ".3f", "expected_pure": ".1f", "observed_pure": ".1f", "f1_09": ".3f", "f1_05": ".3f"}), "",
          "![Pair-level reliability](uncertainty_calibration.png)"]
    (RESULTS / "uncertainty.md").write_text("\n".join(md), encoding="utf-8")
    plot(runs)
    print("\n".join(md))


if __name__ == "__main__":
    main()
