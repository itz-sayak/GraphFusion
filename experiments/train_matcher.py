"""Train the learned column matcher (backend/matching/learned.py).

Training data: Valentine-style fabricated scenarios generated with seed 7777,
which is disjoint from the evaluation benchmark (seed 2026, run_ablation.py).
Neither the sample scenario nor NYC data is used, so both stay held-out.

For every scenario, every candidate column pair is scored with the full
weighted pipeline (so graph-refinement features exist), converted to the
feature vector φ, and labelled from the fabricator's exact ground truth.
Cross-validation is grouped by scenario so no scenario appears in both train
and test folds.

    python experiments/train_matcher.py [--per-level 12]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, save_json
from fabricator import make_benchmark
from run_ablation import profile_files

from backend.config import load_config
from backend.matching.learned import DEFAULT_MODEL_PATH, FEATURES, LearnedMatcher, pair_features
from backend.matching.matcher import MatchStrategy, match_schemas
from backend.storage.workspace import Workspace

TRAIN_SEED = 7777


def build_dataset(per_level: int, cfg: dict, tmp: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, groups = [], [], []
    for i, sc in enumerate(make_benchmark(tmp / "bench", per_level=per_level, seed=TRAIN_SEED)):
        profiles, sketches = profile_files(sc.files, Workspace(tmp / "ws", sc.scenario_id), cfg)
        res = match_schemas(profiles, sketches, cfg, MatchStrategy(scorer="weighted"))
        for (ka, kb), ev in res.evidence.items():
            X.append(pair_features(ev, res.columns[ka], res.columns[kb]))
            y.append(int(tuple(sorted((ka, kb))) in sc.truth))
            groups.append(i)
        print(f"  {sc.scenario_id} ({sc.difficulty}): {len(res.evidence)} pairs, {sum(1 for k in res.evidence if tuple(sorted(k)) in sc.truth)} true")
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-level", type=int, default=12)
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()
    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="dfg_train_"))
    try:
        X, y, groups = build_dataset(args.per_level, cfg, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    model = LearnedMatcher(metadata={"training_seed": TRAIN_SEED, "scenarios": int(len(set(groups))), "per_level": args.per_level})
    model.fit(X, y, groups=groups)
    model.save(Path(args.out), training_data=(X, y))
    # which evidence matters: permutation importance on the training data (for documentation only)
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(model.model, X, y, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = sorted(({"feature": f, "importance": round(float(m), 4)} for f, m in zip(FEATURES, imp.importances_mean)), key=lambda r: -r["importance"])
    save_json("learned_matcher.json", {"metadata": model.metadata, "permutation_importance_auc": importance})
    print(f"\nexamples {len(y)} (positives {int(y.sum())}), out-of-fold AUC {model.metadata['oof_auc']}, Brier {model.metadata['oof_brier']}")
    print("top features:", ", ".join(f"{r['feature']} ({r['importance']})" for r in importance[:6]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
