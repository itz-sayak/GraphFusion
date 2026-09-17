"""Value sketches for scalable overlap estimation.

Each column keeps a **coordinated bottom-k sample** of its distinct values:
DuckDB orders distinct normalised values by ``hash(value)`` and keeps the k
smallest. Because every column uses the same hash function, two samples are
*coordinated* — a value present in both columns is either kept by both or
dropped by both (below a common hash cut-off). This makes Jaccard and
containment estimation from the samples unbiased (KMV / coordinated sampling,
the idea behind LSH Ensemble and JOSIE's sketches), while memory stays O(k)
per column regardless of row count.

When a column has ≤ k distinct values the sample is the complete value set and
all overlap figures are exact.

A MinHash signature is also derived from the sample for LSH candidate
generation across many columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from datasketch import MinHash

from backend.core.values import basic_normalize


@dataclass
class ColumnSketch:
    column: str
    values: dict[str, int]  # basic-normalised value -> coordinated hash
    complete: bool  # True when values holds every distinct value
    tau: int  # largest hash retained (cut-off); meaningful when not complete
    distinct_estimate: int
    minhash: MinHash | None = None
    _normalized_cache: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def normalized(self, name: str, fn: Callable[[Any], str | None]) -> set[str]:
        if name not in self._normalized_cache:
            out = set()
            for v in self.values:
                nv = fn(v)
                if nv is not None:
                    out.add(nv)
            self._normalized_cache[name] = out
        return self._normalized_cache[name]

    def restricted(self, tau: int) -> dict[str, int]:
        return {v: h for v, h in self.values.items() if h <= tau}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_normalized_cache"] = {}
        return state


def build_minhash(values: set[str] | dict[str, int], num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    if values:
        mh.update_batch([v.encode("utf-8") for v in values])
    return mh


def overlap_stats(a: ColumnSketch, b: ColumnSketch, normalizer: str = "basic", fn: Callable[[Any], str | None] | None = None) -> dict[str, float]:
    """Jaccard and two-way containment between two columns under a normaliser.

    For incomplete (sampled) sketches both samples are first restricted to the
    common hash cut-off so the estimate is taken over a coordinated sample.
    Note: hash coordination is defined on basic-normalised values; stronger
    normalisers can merge distinct basic values and make sampled estimates
    approximate (exact whenever both sketches are complete).
    """
    va, vb = a.values, b.values
    if not (a.complete and b.complete):
        tau = min(a.tau if not a.complete else 2**64, b.tau if not b.complete else 2**64)
        va, vb = a.restricted(tau), b.restricted(tau)
    if normalizer == "basic" or fn is None:
        sa, sb = set(va), set(vb)
    elif a.complete and b.complete:
        # normalised value sets are cached per column, so each normaliser runs once per column, not once per pair
        sa, sb = a.normalized(normalizer, fn), b.normalized(normalizer, fn)
    else:
        sa = {x for x in (fn(v) for v in va) if x is not None}
        sb = {x for x in (fn(v) for v in vb) if x is not None}
    if not sa or not sb:
        return {"jaccard": 0.0, "containment_left": 0.0, "containment_right": 0.0, "intersection": 0}
    inter = len(sa & sb)
    return {
        "jaccard": inter / len(sa | sb),
        "containment_left": inter / len(sa),
        "containment_right": inter / len(sb),
        "intersection": inter,
    }


def normalize_for_sketch(value: Any) -> str | None:
    return basic_normalize(value)
