"""Datatype and semantic-type compatibility."""
from __future__ import annotations

from backend.core.models import DataType, SemanticType

_DT = DataType
_COMPAT: dict[frozenset, float] = {
    frozenset({_DT.INTEGER, _DT.FLOAT}): 0.8,
    frozenset({_DT.INTEGER, _DT.STRING}): 0.5,  # ids stored as text
    frozenset({_DT.FLOAT, _DT.STRING}): 0.4,  # amounts stored as '₹1,200'
    frozenset({_DT.DATE, _DT.DATETIME}): 0.9,
    frozenset({_DT.STRING, _DT.DATE}): 0.6,
    frozenset({_DT.STRING, _DT.DATETIME}): 0.6,
    frozenset({_DT.BOOLEAN, _DT.INTEGER}): 0.4,
    frozenset({_DT.BOOLEAN, _DT.STRING}): 0.3,
}


def datatype_similarity(a: DataType, b: DataType) -> float:
    if a == b:
        return 1.0
    return _COMPAT.get(frozenset({a, b}), 0.1)


_ST = SemanticType
_SEM_GROUPS = [
    {_ST.DATE, _ST.DATETIME},
    {_ST.ID, _ST.NUMERIC},
    {_ST.CITY, _ST.CATEGORY, _ST.COUNTRY},
    {_ST.NAME, _ST.FREE_TEXT, _ST.CATEGORY},
    {_ST.CURRENCY, _ST.NUMERIC},
    {_ST.ZIPCODE, _ST.ID, _ST.NUMERIC},
    {_ST.PHONE, _ST.ID},
]
_NEUTRAL = {_ST.UNKNOWN}


def semantic_type_compatibility(a: SemanticType, b: SemanticType) -> float:
    """1 = same, 0.6 = compatible family, 0.5 = unknown (no evidence), 0 = conflicting."""
    if a == b:
        return 1.0
    if a in _NEUTRAL or b in _NEUTRAL:
        return 0.5
    if any(a in g and b in g for g in _SEM_GROUPS):
        return 0.6
    return 0.0
