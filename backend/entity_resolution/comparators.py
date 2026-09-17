"""Field comparators producing discrete agreement levels for Fellegi–Sunter.

Each comparator maps a value pair to (level, similarity). Levels are ordered
from strongest agreement to disagreement; ``null`` means "no evidence" and
contributes a neutral match weight.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from backend.core.models import SemanticType
from backend.core.values import basic_normalize, digits_normalize, get_normalizer, parse_currency, parse_date

NULL = "null"
_TITLE = re.compile(r"\b(mr|mrs|ms|dr|prof)\.?\s+", re.I)


def normalize_person_name(v: Any) -> str | None:
    s = basic_normalize(v)
    if s is None:
        return None
    s = _TITLE.sub("", s)
    if "," in s:  # "sharma, rahul" -> "rahul sharma"
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    tokens = re.sub(r"[^\w\s]", " ", s).split()
    return " ".join(tokens) or None


def name_compare(a: Any, b: Any) -> tuple[str, float]:
    """Person-name agreement.

    exact  : identical after normalisation
    high   : identical ignoring middle initials / token order ("Rahul K Sharma")
    medium : initial-compatible ("R. Sharma") or small typos in both first and last name
    different : otherwise — notably "Rahul Sharma" vs "Rohan Sharma", which a
                whole-string Jaro–Winkler would wrongly score ≈ 0.9
    """
    na, nb = normalize_person_name(a), normalize_person_name(b)
    if not na or not nb:
        return NULL, 0.0
    sim = max(JaroWinkler.similarity(na, nb), fuzz.token_set_ratio(na, nb) / 100)
    if na == nb:
        return "exact", 1.0
    ta, tb = na.split(), nb.split()
    core_a = [t for t in ta if len(t) > 1]
    core_b = [t for t in tb if len(t) > 1]
    if core_a and core_b and sorted(core_a) == sorted(core_b):
        return "high", max(sim, 0.97)
    if len(ta) >= 2 and len(tb) >= 2:
        first_a, first_b, last_a, last_b = ta[0], tb[0], ta[-1], tb[-1]
        last_ok = last_a == last_b or JaroWinkler.similarity(last_a, last_b) >= 0.93
        if last_ok and (len(first_a) == 1 or len(first_b) == 1) and first_a[0] == first_b[0]:
            return "medium", sim
        if last_ok and len(first_a) > 1 and len(first_b) > 1 and JaroWinkler.similarity(first_a, first_b) >= 0.93 and first_a[0] == first_b[0]:
            return "medium", sim
    return "different", sim


def string_compare(normalizer: str = "basic") -> Callable[[Any, Any], tuple[str, float]]:
    fn = get_normalizer(normalizer)

    def _cmp(a: Any, b: Any) -> tuple[str, float]:
        na, nb = fn(a), fn(b)
        if not na or not nb:
            return NULL, 0.0
        if na == nb:
            return "exact", 1.0
        sim = JaroWinkler.similarity(na, nb)
        if sim >= 0.94:
            return "high", sim
        if sim >= 0.85:
            return "medium", sim
        return "different", sim

    return _cmp


def exact_compare(normalizer: str = "basic") -> Callable[[Any, Any], tuple[str, float]]:
    fn = get_normalizer(normalizer)

    def _cmp(a: Any, b: Any) -> tuple[str, float]:
        na, nb = fn(a), fn(b)
        if not na or not nb:
            return NULL, 0.0
        return ("exact", 1.0) if na == nb else ("different", 0.0)

    return _cmp


def phone_compare(a: Any, b: Any) -> tuple[str, float]:
    na, nb = digits_normalize(a), digits_normalize(b)
    if not na or not nb:
        return NULL, 0.0
    if na == nb:
        return "exact", 1.0
    if na[-7:] == nb[-7:]:
        return "high", 0.9
    return "different", 0.0


def date_compare(a: Any, b: Any) -> tuple[str, float]:
    da, db = parse_date(a, dayfirst=True), parse_date(b, dayfirst=True)
    if da is None or db is None:
        return NULL, 0.0
    days = abs((da - db).days)
    if days == 0:
        return "exact", 1.0
    if days <= 31:
        return "high", 0.8
    return "different", 0.0


def numeric_compare(a: Any, b: Any) -> tuple[str, float]:
    xa, _ = parse_currency(a)
    xb, _ = parse_currency(b)
    if xa is None or xb is None:
        return NULL, 0.0
    denom = max(abs(xa), abs(xb), 1e-9)
    rel = abs(xa - xb) / denom
    if rel <= 0.001:
        return "exact", 1.0
    if rel <= 0.05:
        return "high", 1 - rel
    return "different", max(0.0, 1 - rel)


def comparator_for(semantic_type: SemanticType, normalizer: str = "basic") -> tuple[str, Callable[[Any, Any], tuple[str, float]]]:
    st = semantic_type
    if st == SemanticType.NAME:
        return "person_name", name_compare
    if st == SemanticType.EMAIL:
        return "email_exact", exact_compare("basic")
    if st == SemanticType.PHONE:
        return "phone_digits", phone_compare
    if st in (SemanticType.DATE, SemanticType.DATETIME):
        return "date_window", date_compare
    if st in (SemanticType.CURRENCY, SemanticType.NUMERIC):
        return "numeric_relative", numeric_compare
    if st in (SemanticType.ID, SemanticType.ZIPCODE):
        return f"exact[{normalizer}]", exact_compare(normalizer if normalizer in ("identifier", "basic", "alnum", "digits") else "identifier")
    if st in (SemanticType.CITY, SemanticType.COUNTRY, SemanticType.CATEGORY):
        return f"exact[{normalizer}]", exact_compare(normalizer)
    return f"string[{normalizer}]", string_compare(normalizer)


LEVELS = ["exact", "high", "medium", "different"]
