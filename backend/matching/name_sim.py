"""Lexical column-name similarity.

The composite takes the best of three views, each robust to different drift:

* weighted, symmetric **Monge–Elkan** over abbreviation-expanded tokens with
  Jaro–Winkler as the inner measure (handles re-ordering and partial tokens);
  generic tokens such as ``identifier`` or ``name`` carry half weight so that
  ``customer_id`` vs ``product_id`` is not scored as a near match;
* **Jaro–Winkler** on the joined normalised form (prefix-heavy drift);
* **token-set ratio** (Levenshtein-based, order-insensitive).
"""
from __future__ import annotations

from functools import lru_cache

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from backend.matching.normalize import name_tokens

GENERIC_TOKENS = {"identifier", "number", "name", "code", "date", "amount", "value", "type", "count", "total", "time", "key", "description", "flag", "status"}


def _weight(tok: str) -> float:
    return 0.5 if tok in GENERIC_TOKENS else 1.0


def _monge_elkan(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    num = den = 0.0
    for ta in a:
        best = max(JaroWinkler.similarity(ta, tb) for tb in b)
        # inner similarities below 0.8 are mostly noise between unrelated words
        best = best if best >= 0.8 else best * 0.5
        w = _weight(ta)
        num += w * best
        den += w
    return num / den


@lru_cache(maxsize=65536)
def name_similarity(a: str, b: str, vendor_prefixes: tuple[str, ...] = ()) -> float:
    ta, tb = name_tokens(a, vendor_prefixes), name_tokens(b, vendor_prefixes)
    ja, jb = "_".join(ta), "_".join(tb)
    if not ja or not jb:
        return 0.0
    if ja == jb:
        return 1.0
    me = (_monge_elkan(ta, tb) + _monge_elkan(tb, ta)) / 2
    jw = JaroWinkler.similarity(ja, jb)
    tsr = fuzz.token_set_ratio(" ".join(ta), " ".join(tb)) / 100
    return round(max(me, 0.9 * jw if jw > 0.85 else 0.6 * jw, 0.95 * tsr if tsr > 0.8 else 0.6 * tsr), 4)


def raw_levenshtein(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a.lower(), b.lower())
