"""Candidate column-pair generation (blocking for schema matching).

Exhaustive scoring is O(|C|²) in the number of columns. For small sessions
that is cheapest; beyond ``lsh_pair_threshold`` pairs we generate candidates
from the union of three sub-linear indexes:

* **MinHash LSH Ensemble** (Zhu et al., VLDB'16) over value sketches, queried
  by *containment* rather than Jaccard, so a 265-value dimension key is still
  found from a 20 000-value foreign-key column;
* an inverted index on abbreviation-expanded **name tokens** / thesaurus concepts;
* an index on confident **semantic types** (EMAIL, PHONE, DATE, …).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from datasketch import MinHashLSHEnsemble

from backend.core.models import SemanticType
from backend.matching.normalize import concepts, name_tokens
from backend.matching.name_sim import GENERIC_TOKENS
from backend.matching.scorer import ColumnContext

_SPECIFIC_TYPES = {
    SemanticType.EMAIL, SemanticType.PHONE, SemanticType.DATE, SemanticType.DATETIME, SemanticType.CITY,
    SemanticType.COUNTRY, SemanticType.ZIPCODE, SemanticType.LATITUDE, SemanticType.LONGITUDE, SemanticType.NAME,
}


def all_cross_pairs(columns: list[ColumnContext]) -> set[tuple[str, str]]:
    return {tuple(sorted((a.key, b.key))) for a, b in combinations(columns, 2) if a.dataset_id != b.dataset_id}


def candidate_pairs(columns: list[ColumnContext], config: dict) -> tuple[set[tuple[str, str]], str]:
    sm = config["schema_matching"]
    by_ds: dict[str, int] = defaultdict(int)
    for c in columns:
        by_ds[c.dataset_id] += 1
    total_pairs = (sum(by_ds.values()) ** 2 - sum(v * v for v in by_ds.values())) // 2
    if total_pairs <= sm["lsh_pair_threshold"]:
        return all_cross_pairs(columns), "exhaustive"

    ctx = {c.key: c for c in columns}
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a != b and ctx[a].dataset_id != ctx[b].dataset_id:
            pairs.add(tuple(sorted((a, b))))

    # 1) containment LSH over value sketches
    lsh = MinHashLSHEnsemble(threshold=sm["lsh_threshold"], num_perm=sm["minhash_permutations"], num_part=16)
    entries = [(c.key, c.sketch.minhash, max(1, len(c.sketch.values))) for c in columns if c.sketch.minhash is not None and c.sketch.values]
    if entries:
        lsh.index(entries)
        for key, mh, size in entries:
            for other in lsh.query(mh, size):
                add(key, other)

    # 2) name-token / concept inverted index (generic tokens excluded)
    inverted: dict[str, list[str]] = defaultdict(list)
    for c in columns:
        toks = [t for t in name_tokens(c.profile.name) if t not in GENERIC_TOKENS]
        feats = set(toks) | {f"concept:{x}" for x in concepts(toks)}
        for f in feats:
            inverted[f].append(c.key)
    for keys in inverted.values():
        if len(keys) <= 200:
            for a, b in combinations(keys, 2):
                add(a, b)

    # 3) confident specific semantic types
    by_type: dict[SemanticType, list[str]] = defaultdict(list)
    for c in columns:
        if c.profile.semantic_type in _SPECIFIC_TYPES and c.profile.semantic_confidence >= 0.6:
            by_type[c.profile.semantic_type].append(c.key)
    for keys in by_type.values():
        for a, b in combinations(keys, 2):
            add(a, b)
    return pairs, "lsh_ensemble+name+semantic"
