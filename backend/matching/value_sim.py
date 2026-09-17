"""Instance-based (value) similarity with data-driven normaliser selection.

For a column pair we try every *applicable* registered normaliser (basic,
alnum, identifier, geo, reference-domain maps such as Bengaluru↔Bangalore)
and keep the one that maximises overlap. The winning normaliser is returned
as evidence and later re-used verbatim as the merge transformation, so the
join is executed under exactly the equivalence that justified it.
"""
from __future__ import annotations

import math

from backend.core.models import ColumnProfile, DataType, SemanticType
from backend.core.values import get_normalizer, normalizers
from backend.profiling.sketches import ColumnSketch, overlap_stats

_TEXTUAL = {SemanticType.CITY, SemanticType.COUNTRY, SemanticType.CATEGORY, SemanticType.NAME, SemanticType.UNKNOWN, SemanticType.FREE_TEXT}
# tie-break towards simpler normalisers; composite ones rank after a single domain
_NORMALIZER_COST = {"basic": 0.0, "alnum": 0.002, "identifier": 0.004, "digits": 0.004, "geo": 0.006}
_DEFAULT_COST = 0.008


def applicable_normalizers(pa: ColumnProfile, pb: ColumnProfile) -> list[str]:
    names = ["basic", "alnum"]
    numeric_like = {DataType.INTEGER, DataType.FLOAT}
    if pa.semantic_type in (SemanticType.ID, SemanticType.ZIPCODE) or pb.semantic_type in (SemanticType.ID, SemanticType.ZIPCODE) or (
        pa.data_type in numeric_like or pb.data_type in numeric_like
    ):
        names.append("identifier")
    if SemanticType.PHONE in (pa.semantic_type, pb.semantic_type):
        names.append("digits")
    if pa.data_type == DataType.STRING and pb.data_type == DataType.STRING and (pa.semantic_type in _TEXTUAL and pb.semantic_type in _TEXTUAL):
        names.append("geo")
        names += [n for n in normalizers() if n.startswith("domain:") or n.startswith("semantic:")]
    return names


_PREFIXED = __import__("re").compile(r"^([A-Za-z]+)[-_ ]?\d+$")


def _code_prefixes(sketch: ColumnSketch, limit: int = 300) -> set[str] | None:
    """Alphabetic prefixes of coded identifiers ('OR-0000123' → 'OR'); None unless ≥ 80% of values are coded."""
    vals = list(sketch.values)[:limit]
    if not vals:
        return None
    prefixes = [m.group(1).upper() for v in vals if (m := _PREFIXED.match(v.strip()))]
    return set(prefixes) if len(prefixes) >= 0.8 * len(vals) else None


def _integers(sketch: ColumnSketch) -> list[int] | None:
    out = []
    for v in sketch.values:
        s = v.lstrip("-")
        if not s.isdigit():
            return None
        out.append(int(v))
    return out


def expected_containment(source: list[int], target: list[int]) -> float:
    """Containment of ``source`` in ``target`` expected by chance: the density of
    ``target`` within the value range spanned by ``source``. A dense key such as
    LocationID = 1..265 contains *any* small-integer column, so observing that
    containment carries no evidence."""
    lo, hi = min(source), max(source)
    width = hi - lo + 1
    inside = sum(1 for t in target if lo <= t <= hi)
    return min(1.0, inside / width) if width > 0 else 1.0


def _all_numeric(sketch: ColumnSketch, limit: int = 200) -> bool:
    vals = list(sketch.values)[:limit]
    return bool(vals) and all(v.lstrip("-").replace(".", "", 1).isdigit() for v in vals)


def value_similarity(pa: ColumnProfile, sa: ColumnSketch, pb: ColumnProfile, sb: ColumnSketch, extra_normalizers: tuple[str, ...] = ()) -> dict[str, float | str]:
    if not sa.values or not sb.values:
        return {"value_overlap": 0.0, "jaccard": 0.0, "containment_left": 0.0, "containment_right": 0.0, "normalizer": "basic"}
    best: dict | None = None
    best_key = -1.0
    # the identifier normaliser strips code prefixes ('C-00123' ≡ '123'); when *both* columns are coded with
    # disjoint prefixes ('OR-…' vs 'PR-…') they are different identifier systems, and stripping would fake overlap
    pre_a, pre_b = _code_prefixes(sa), _code_prefixes(sb)
    skip_identifier = pre_a is not None and pre_b is not None and not (pre_a & pre_b)
    for name in [*applicable_normalizers(pa, pb), *extra_normalizers]:
        if name == "identifier" and skip_identifier:
            continue
        stats = overlap_stats(sa, sb, name, get_normalizer(name))
        # choose by the same objective that is reported as V (before the domain-size discount)
        key = 0.4 * stats["jaccard"] + 0.6 * max(stats["containment_left"], stats["containment_right"]) - _NORMALIZER_COST.get(name, _DEFAULT_COST + (0.001 if name.startswith("semantic:") else 0.0))
        if key > best_key + 1e-9:
            best_key, best = key, {**stats, "normalizer": name}
    assert best is not None
    cont = max(best["containment_left"], best["containment_right"])
    raw = 0.4 * best["jaccard"] + 0.6 * cont

    # Small value domains overlap by coincidence ({1..6} ⊂ {1..265}); discount
    # them, strongly for purely numeric domains and mildly for textual ones.
    smaller = min(len(sa.values), len(sb.values))
    distinct_factor = min(1.0, math.log2(1 + smaller) / math.log2(1 + 32))
    notes = ""
    ia, ib = (_integers(sa), _integers(sb)) if sa.complete and sb.complete else (None, None)
    if ia and ib:
        # integer domains: keep only the overlap that is *surprising* given the value ranges
        adj = []
        for src, tgt, cont in ((ia, ib, best["containment_left"]), (ib, ia, best["containment_right"])):
            e = expected_containment(src, tgt)
            adj.append(max(0.0, (cont - e) / (1 - e)) if e < 0.999 else 0.0)
        raw = 0.5 * raw + 0.5 * max(adj)
        factor = distinct_factor
        notes = f"integer domains: chance-adjusted containment {max(adj):.2f}"
    elif _all_numeric(sa) and _all_numeric(sb):
        factor = distinct_factor
        if pa.data_type == DataType.FLOAT and pb.data_type == DataType.FLOAT:
            factor *= 0.5  # equal measurements (10.4 vs 10.4) are weak evidence of the same attribute
            notes = "continuous measures: coincidental value equality discounted"
    else:
        factor = 0.8 + 0.2 * distinct_factor
    return {
        "note": notes,
        "value_overlap": round(raw * factor, 4),
        "jaccard": round(best["jaccard"], 4),
        "containment_left": round(best["containment_left"], 4),
        "containment_right": round(best["containment_right"], 4),
        "normalizer": best["normalizer"],
    }
