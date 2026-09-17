"""Learned column matcher.

The weighted score S(ci, cj) combines evidence with hand-set weights. The
learned matcher instead estimates

    P(match | φ(ci, cj))

with a gradient-boosted tree ensemble over the same evidence features φ
(the seven signals, overlap statistics, graph-refinement quantities and column
statistics), calibrated with isotonic regression so the output can be used
with the existing HIGH/MEDIUM thresholds.

Training data comes from the fabricated benchmark generator with seeds
disjoint from the evaluation benchmark (see ``experiments/train_matcher.py``).
User approve/reject decisions are added as extra, up-weighted examples when
the matcher is retrained inside a session (a simple human-in-the-loop update).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import PROJECT_ROOT
from backend.core.models import ColumnProfile, DataType, MatchEvidence
from backend.matching.type_sim import semantic_type_compatibility

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "column_matcher.joblib"

FEATURES = [
    "name_similarity", "datatype_similarity", "semantic_similarity", "value_overlap", "distribution_similarity",
    "pattern_similarity", "cardinality_similarity", "jaccard", "containment_max", "containment_min",
    "base_score", "transitive_support", "structural_support", "exclusivity",
    "semantic_type_compat", "uniqueness_max", "uniqueness_min", "distinct_ratio", "both_numeric", "same_semantic_type",
]
_NUMERIC = {DataType.INTEGER, DataType.FLOAT}


def pair_features(ev: MatchEvidence, a: ColumnProfile, b: ColumnProfile) -> list[float]:
    da, db = max(a.unique_count, 1), max(b.unique_count, 1)
    return [
        ev.name_similarity, ev.datatype_similarity, ev.semantic_similarity, ev.value_overlap, ev.distribution_similarity,
        ev.pattern_similarity, ev.cardinality_similarity, ev.jaccard,
        max(ev.containment_left, ev.containment_right), min(ev.containment_left, ev.containment_right),
        ev.base_score, ev.transitive_support, ev.structural_support, ev.exclusivity,
        semantic_type_compatibility(a.semantic_type, b.semantic_type),
        max(a.uniqueness, b.uniqueness), min(a.uniqueness, b.uniqueness),
        min(da, db) / max(da, db),
        float(a.data_type in _NUMERIC and b.data_type in _NUMERIC),
        float(a.semantic_type == b.semantic_type),
    ]


@dataclass
class LearnedMatcher:
    model: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, groups: np.ndarray | None = None, seed: int = 0) -> "LearnedMatcher":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.model_selection import GroupKFold, StratifiedKFold

        base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=seed)
        # out-of-fold estimate of quality before fitting on everything (grouped by scenario to avoid leakage)
        oof = np.zeros(len(y))
        splitter = GroupKFold(n_splits=5) if groups is not None and len(set(groups)) >= 5 else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups) if groups is not None and len(set(groups)) >= 5 else splitter.split(X, y)
        for train_idx, test_idx in split_iter:
            fold = CalibratedClassifierCV(_clone(base), method="isotonic", cv=3)
            fold.fit(X[train_idx], y[train_idx], sample_weight=None if sample_weight is None else sample_weight[train_idx])
            oof[test_idx] = fold.predict_proba(X[test_idx])[:, 1]
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.metadata.update({
            "features": FEATURES,
            "examples": int(len(y)),
            "positives": int(y.sum()),
            "cv": "GroupKFold(5) by scenario" if groups is not None else "StratifiedKFold(5)",
            "oof_auc": round(float(roc_auc_score(y, oof)), 4) if 0 < y.sum() < len(y) else None,
            "oof_brier": round(float(brier_score_loss(y, oof)), 4),
        })
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("learned matcher is not trained")
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = DEFAULT_MODEL_PATH, training_data: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "metadata": self.metadata, "training_data": training_data}, path)
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "LearnedMatcher":
        import joblib

        blob = joblib.load(path)
        m = cls(blob["model"], blob["metadata"])
        m._training_data = blob.get("training_data")  # type: ignore[attr-defined]
        return m

    def retrain_with_decisions(self, extra_X: np.ndarray, extra_y: np.ndarray, weight: float = 10.0) -> "LearnedMatcher":
        """Refit on the stored base training set plus user-labelled pairs (up-weighted)."""
        base = getattr(self, "_training_data", None)
        if base is None:
            raise RuntimeError("the saved model does not include its training data; retrain with experiments/train_matcher.py")
        X0, y0 = base
        X = np.vstack([X0, extra_X])
        y = np.concatenate([y0, extra_y])
        w = np.concatenate([np.ones(len(y0)), np.full(len(extra_y), weight)])
        new = LearnedMatcher(metadata={**self.metadata, "user_labels": int(len(extra_y)), "user_label_weight": weight})
        new.fit(X, y, sample_weight=w)
        new._training_data = base  # type: ignore[attr-defined]
        return new


def _clone(est):
    from sklearn.base import clone

    return clone(est)


_CACHE: dict[str, LearnedMatcher] = {}


def load_default(path: Path | None = None) -> LearnedMatcher | None:
    p = Path(path or DEFAULT_MODEL_PATH)
    if not p.exists():
        return None
    key = str(p.resolve())
    if key not in _CACHE:
        _CACHE[key] = LearnedMatcher.load(p)
    return _CACHE[key]
