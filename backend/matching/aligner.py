"""Constraint-aware schema alignment via maximum-weight bipartite matching.

Thresholding pairwise scores independently lets one column match many
(``name`` with ``first_name`` *and* ``last_name`` *and* ``full_name``). For each
dataset pair we instead solve the assignment problem on the score matrix with
the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``), which
yields the globally optimal 1:1 alignment.

Many-to-one is legitimate for foreign keys (``PULocationID`` and
``DOLocationID`` both reference ``LocationID``). A non-assigned pair is
re-admitted as ``foreign_key_candidate`` when the target column is a unique
key and the other column's values are contained in it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.core.models import ColumnMatch


def is_fk_reference(m: ColumnMatch, uniqueness: dict[str, float], key_uniqueness: float, min_containment: float) -> str | None:
    """Return 'right'/'left' when that side is a unique key containing the other side's values."""
    ev = m.evidence
    if uniqueness.get(m.right.key, 0) >= key_uniqueness and ev.containment_left >= min_containment:
        return "right"
    if uniqueness.get(m.left.key, 0) >= key_uniqueness and ev.containment_right >= min_containment:
        return "left"
    return None


def align(
    matches: list[ColumnMatch],
    min_score: float,
    key_uniqueness: float,
    fk_min_containment: float,
    uniqueness: dict[str, float],
    allow_fk: bool = True,
    one_to_one: bool = True,
    vetoed: dict[tuple[str, str], str] | None = None,
) -> list[ColumnMatch]:
    """``vetoed``: pairs that must not be accepted whatever their score (reason per pair); they are excluded
    *before* the 1:1 assignment so they cannot take a column away from its true partner."""
    vetoed = vetoed or {}
    by_pair: dict[tuple[str, str], list[ColumnMatch]] = defaultdict(list)
    for m in matches:
        by_pair[(m.left.dataset_id, m.right.dataset_id)].append(m)

    for group in by_pair.values():
        for m in group:
            m.accepted = False
            m.rejection_reason = None
        eligible = [m for m in group if m.score >= min_score and m.pair_key not in vetoed]
        if eligible and not one_to_one:
            for m in eligible:
                m.accepted = True
        elif eligible:
            lefts = sorted({m.left.column for m in eligible})
            rights = sorted({m.right.column for m in eligible})
            li = {c: i for i, c in enumerate(lefts)}
            ri = {c: i for i, c in enumerate(rights)}
            mat = np.zeros((len(lefts), len(rights)))
            lookup: dict[tuple[int, int], ColumnMatch] = {}
            for m in eligible:
                i, j = li[m.left.column], ri[m.right.column]
                mat[i, j] = m.score
                lookup[(i, j)] = m
            rows, cols = linear_sum_assignment(mat, maximize=True)
            for i, j in zip(rows, cols):
                if (i, j) in lookup:
                    lookup[(i, j)].accepted = True
            if allow_fk:
                for m in eligible:
                    if not m.accepted and is_fk_reference(m, uniqueness, key_uniqueness, fk_min_containment):
                        m.accepted = True
                        m.evidence.notes.append("re-admitted after 1:1 assignment: many-to-one reference into a unique key")
        for m in group:
            if m.accepted and is_fk_reference(m, uniqueness, key_uniqueness, fk_min_containment):
                m.relationship = "foreign_key_candidate"
            if not m.accepted and m.pair_key in vetoed:
                m.rejection_reason = vetoed[m.pair_key]
            elif not m.accepted:
                m.rejection_reason = (
                    f"score {m.score:.2f} below alignment threshold {min_score:.2f}"
                    if m.score < min_score
                    else "lost the 1:1 assignment to a higher-scoring alternative"
                )
    return matches
