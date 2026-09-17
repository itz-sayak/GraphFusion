"""Conflict resolution strategies for data fusion (Bleiholder & Naumann, 2008).

A *conflict* exists when the records of one entity carry values for the same
canonical attribute that differ after the attribute's normaliser is applied
('Bangalore' vs 'Bengaluru' under ``domain:indian_cities`` is not a conflict;
'Mumbai' vs 'Pune' is). Nothing is overwritten silently: every conflict is
recorded with all candidates and their sources, whatever the strategy picks.

Strategies
----------
majority_vote        most frequent normalised value; ties → source priority
prefer_source        first non-null value in dataset priority order
prefer_latest        value from the most recently updated record (record timestamp), else majority
prefer_non_null      first non-null value in record order (conflict still flagged)
highest_confidence   value from the record with the highest match confidence
source_accuracy_vote truth discovery: iteratively estimate per-source accuracy and weight votes
                     by log(acc / (1 − acc)) (ACCU-style, Dong et al. VLDB'09)
keep_all             all distinct values, JSON-encoded
manual_review        no value is chosen (NULL) and the conflict is queued for review
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Candidate:
    value: Any
    norm: str | None
    dataset: str
    column: str
    row: int
    confidence: float = 1.0
    timestamp: datetime | None = None
    priority: int = 0


@dataclass
class Resolution:
    value: Any
    reason: str
    status: str  # resolved | flagged | manual_review
    winner: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _majority(cands: list[Candidate]) -> tuple[Candidate, str]:
    counts = Counter(c.norm for c in cands)
    top = max(counts.values())
    tied = {n for n, k in counts.items() if k == top}
    winner = min((c for c in cands if c.norm in tied), key=lambda c: (c.priority, c.row))
    reason = f"majority vote ({top} of {len(cands)} records)" if len(tied) == 1 else f"tie between {len(tied)} values at {top} vote(s); broken by source priority ({winner.dataset})"
    return winner, reason


def resolve(cands: list[Candidate], strategy: str, source_accuracy: dict[str, float] | None = None) -> Resolution:
    present = [c for c in cands if c.norm is not None]
    if not present:
        return Resolution(None, "no non-null values", "resolved", None, cands)
    distinct = {c.norm for c in present}
    if len(distinct) == 1:
        winner = min(present, key=lambda c: (c.priority, c.row))
        return Resolution(winner.value, "all sources agree", "resolved", winner, cands)

    if strategy == "majority_vote":
        w, reason = _majority(present)
    elif strategy == "prefer_source":
        w = min(present, key=lambda c: (c.priority, c.row))
        reason = f"preferred source {w.dataset}"
    elif strategy == "prefer_latest":
        dated = [c for c in present if c.timestamp is not None]
        if dated:
            w = max(dated, key=lambda c: (c.timestamp, -c.priority))
            reason = f"most recent record ({w.dataset} updated {w.timestamp:%Y-%m-%d %H:%M})"
        else:
            w, reason = _majority(present)
            reason = "no record timestamps available; fell back to " + reason
    elif strategy == "prefer_non_null":
        w = min(present, key=lambda c: (c.priority, c.row))
        reason = f"first non-null value ({w.dataset})"
    elif strategy == "highest_confidence":
        w = max(present, key=lambda c: (c.confidence, -c.priority))
        reason = f"highest match confidence {w.confidence:.2f} ({w.dataset})"
    elif strategy == "source_accuracy_vote":
        acc = source_accuracy or {}
        scores: dict[str, float] = defaultdict(float)
        for c in present:
            a = min(max(acc.get(c.dataset, 0.8), 0.01), 0.99)
            scores[c.norm] += math.log(a / (1 - a))
        best = max(scores, key=lambda n: scores[n])
        w = min((c for c in present if c.norm == best), key=lambda c: (c.priority, c.row))
        reason = "source-accuracy weighted vote (" + ", ".join(f"{d}={acc.get(d, 0.8):.2f}" for d in sorted({c.dataset for c in present})) + ")"
    elif strategy == "keep_all":
        values = []
        for c in sorted(present, key=lambda c: (c.priority, c.row)):
            if _jsonable(c.value) not in values:
                values.append(_jsonable(c.value))
        return Resolution(json.dumps(values, ensure_ascii=False, default=str), f"kept all {len(values)} distinct values", "flagged", None, cands)
    elif strategy == "manual_review":
        return Resolution(None, f"{len(distinct)} conflicting values queued for manual review", "manual_review", None, cands)
    else:  # pragma: no cover - guarded by the planner
        raise ValueError(strategy)
    return Resolution(w.value, reason, "flagged", w, cands)


def estimate_source_accuracy(conflict_sets: list[list[Candidate]], iterations: int = 10) -> dict[str, float]:
    """Truth discovery over all attributes of all entities at once."""
    datasets = {c.dataset for cs in conflict_sets for c in cs}
    acc = {d: 0.8 for d in datasets}
    for _ in range(iterations):
        agree: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for cs in conflict_sets:
            present = [c for c in cs if c.norm is not None]
            if len({c.norm for c in present}) < 2:
                continue
            truth = resolve(present, "source_accuracy_vote", acc).winner
            for c in present:
                total[c.dataset] += 1
                agree[c.dataset] += int(truth is not None and c.norm == truth.norm)
        new = {d: (agree[d] + 1) / (total[d] + 2) for d in datasets}  # Laplace
        if all(abs(new[d] - acc[d]) < 1e-4 for d in datasets):
            acc = new
            break
        acc = new
    return {d: round(a, 4) for d, a in acc.items()}
