"""Distribution, cardinality and format-pattern similarity."""
from __future__ import annotations

import math

from backend.core.models import ColumnProfile, DataType

_NUMERIC = {DataType.INTEGER, DataType.FLOAT}


def distribution_similarity(a: ColumnProfile, b: ColumnProfile) -> float:
    if a.data_type in _NUMERIC and b.data_type in _NUMERIC and a.quantiles and b.quantiles:
        qa = [q for q in a.quantiles if q is not None]
        qb = [q for q in b.quantiles if q is not None]
        if len(qa) != len(qb) or not qa:
            return 0.0
        lo = min(min(qa), min(qb))
        hi = max(max(qa), max(qb))
        span = hi - lo
        if span <= 0:
            return 1.0
        # mean absolute quantile distance on the pooled range (a 1-D Wasserstein proxy)
        d = sum(abs(x - y) for x, y in zip(qa, qb)) / len(qa) / span
        return round(max(0.0, 1.0 - d), 4)
    if a.avg_length and b.avg_length:
        return round(1.0 - abs(a.avg_length - b.avg_length) / max(a.avg_length, b.avg_length), 4)
    if (a.data_type in _NUMERIC) != (b.data_type in _NUMERIC):
        return 0.2
    return 0.5


def cardinality_similarity(a: ColumnProfile, b: ColumnProfile) -> float:
    ua, ub = a.unique_count, b.unique_count
    if ua == 0 or ub == 0:
        return 0.0
    return round(math.sqrt(min(ua, ub) / max(ua, ub)), 4)


def pattern_similarity(a: ColumnProfile, b: ColumnProfile) -> float:
    ha, hb = a.pattern_histogram, b.pattern_histogram
    if not ha or not hb:
        return 0.0
    keys = set(ha) | set(hb)
    dot = sum(ha.get(k, 0.0) * hb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in ha.values()))
    nb = math.sqrt(sum(v * v for v in hb.values()))
    return round(dot / (na * nb), 4) if na and nb else 0.0
