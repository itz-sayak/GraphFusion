"""Blocking: sub-quadratic candidate generation for entity resolution.

Comparing every record with every other is O(n²). Candidates are the union
of three complementary blocking schemes:

* **standard blocking** on highly discriminative normalised values
  (e-mail, phone digits, shared identifiers) — catches easy matches cheaply;
* **token blocking** on name keys ``<surname>|<first initial>`` — robust to
  middle initials, casing and "R. Sharma" abbreviations;
* **sorted neighbourhood** (Hernández & Stolfo, SIGMOD'95) over a sort key of
  the most descriptive text field with a sliding window — catches typos that
  break exact keys.

**Adaptive refinement.** Name-token keys stay equally coarse as the data grows
(``sharma|r`` collects every R. Sharma), so block sizes, and pairs, grow
quadratically. A token block larger than ``refine_block_size`` is split by a
second key instead of being compared in full: first by the normalised value of a
location/category field (keeps "R. Sharma" and "Rahul Sharma" together), then, if
still too large, by the full first name. Records without the refinement value
form their own sub-block. Blocks larger than ``max_block_size`` after refinement
are skipped (and reported).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from backend.core.models import SemanticType
from backend.core.values import basic_normalize, digits_normalize, get_normalizer
from backend.entity_resolution.comparators import normalize_person_name

Record = tuple[int, int]  # (dataset index, row index)


@dataclass
class BlockingStats:
    schemes: dict[str, int] = field(default_factory=dict)
    skipped_blocks: int = 0
    refined_blocks: int = 0
    candidate_pairs: int = 0
    total_possible_pairs: int = 0

    @property
    def reduction_ratio(self) -> float:
        if not self.total_possible_pairs:
            return 0.0
        return 1 - self.candidate_pairs / self.total_possible_pairs


def _name_key(v: Any) -> str | None:
    n = normalize_person_name(v)
    if not n:
        return None
    toks = n.split()
    if len(toks) == 1:
        return toks[0]
    return f"{toks[-1]}|{toks[0][0]}"


def _first_name(v: Any) -> str | None:
    n = normalize_person_name(v)
    toks = n.split() if n else []
    return toks[0] if len(toks) > 1 and len(toks[0]) > 1 else None


def _refine(members: list[Record], records: dict[Record, dict[str, Any]], key_fns: list, limit: int) -> list[list[Record]]:
    """Split a block by successive secondary keys until every part is within ``limit`` (or keys run out)."""
    parts = [members]
    for fname, fn in key_fns:
        if all(len(p) <= limit for p in parts):
            break
        nxt: list[list[Record]] = []
        for p in parts:
            if len(p) <= limit:
                nxt.append(p)
                continue
            groups: dict[Any, list[Record]] = defaultdict(list)
            for rec in p:
                v = records[rec].get(fname)
                groups[fn(v) if v is not None else None].append(rec)
            nxt.extend(groups.values())
        parts = nxt
    return parts


def block_key_fn(semantic_type: SemanticType, normalizer: str):
    if semantic_type == SemanticType.EMAIL:
        return "standard", basic_normalize
    if semantic_type == SemanticType.PHONE:
        return "standard", digits_normalize
    if semantic_type == SemanticType.NAME:
        return "token", _name_key
    if semantic_type in (SemanticType.ID,):
        return "standard", get_normalizer(normalizer)
    return None, None


def generate_candidates(
    records: dict[Record, dict[str, Any]],
    fields: list[tuple[str, SemanticType, str]],
    allow_within: set[int],
    max_block_size: int,
    window: int,
    schemes: list[str],
    refine_block_size: int | None = None,
) -> tuple[set[tuple[Record, Record]], BlockingStats]:
    """``fields``: (field name, semantic type, normaliser). ``allow_within``: dataset indexes deduplicated internally."""
    stats = BlockingStats()
    refine_fns = []  # secondary keys for oversized token blocks, most recall-preserving first
    for fname, stype, normalizer in fields:
        if stype in (SemanticType.CITY, SemanticType.COUNTRY, SemanticType.CATEGORY, SemanticType.ZIPCODE):
            refine_fns.append((fname, get_normalizer(normalizer) if normalizer else basic_normalize))
    first_name_fns = [(fname, _first_name) for fname, stype, _ in fields if stype == SemanticType.NAME]
    pairs: set[tuple[Record, Record]] = set()

    def admissible(a: Record, b: Record) -> bool:
        if a == b:
            return False
        return a[0] != b[0] or a[0] in allow_within

    def add_block(members: list[Record], scheme: str) -> None:
        if len(members) < 2:
            return
        if len(members) > max_block_size:
            stats.skipped_blocks += 1
            return
        n = 0
        for a, b in combinations(members, 2):
            if admissible(a, b):
                pairs.add((a, b) if a < b else (b, a))
                n += 1
        stats.schemes[scheme] = stats.schemes.get(scheme, 0) + n

    for fname, stype, normalizer in fields:
        scheme, fn = block_key_fn(stype, normalizer)
        if scheme is None or scheme not in schemes:
            continue
        blocks: dict[str, list[Record]] = defaultdict(list)
        for rec, row in records.items():
            key = fn(row.get(fname))
            if key:
                blocks[key].append(rec)
        for members in blocks.values():
            if scheme == "token" and refine_block_size and len(members) > refine_block_size:
                for sub in _refine(members, records, refine_fns + first_name_fns, refine_block_size):
                    add_block(sub, f"{scheme}:{fname}:refined")
                stats.refined_blocks += 1
            else:
                add_block(members, f"{scheme}:{fname}")

    if "sorted_neighborhood" in schemes:
        text_fields = [f for f, st, _ in fields if st == SemanticType.NAME] or [f for f, st, _ in fields if st in (SemanticType.FREE_TEXT, SemanticType.UNKNOWN, SemanticType.CATEGORY)]
        for fname in text_fields[:1]:
            keyed = []
            for rec, row in records.items():
                n = normalize_person_name(row.get(fname))
                if n:
                    toks = n.split()
                    keyed.append((" ".join([toks[-1]] + toks[:-1]), rec))
            keyed.sort()
            n_added = 0
            for i in range(len(keyed)):
                for j in range(i + 1, min(i + window, len(keyed))):
                    a, b = keyed[i][1], keyed[j][1]
                    if admissible(a, b):
                        pairs.add((a, b) if a < b else (b, a))
                        n_added += 1
            stats.schemes[f"sorted_neighborhood:{fname}"] = n_added

    sizes: dict[int, int] = defaultdict(int)
    for rec in records:
        sizes[rec[0]] += 1
    total = 0
    ds = sorted(sizes)
    for i, a in enumerate(ds):
        if a in allow_within:
            total += sizes[a] * (sizes[a] - 1) // 2
        for b in ds[i + 1 :]:
            total += sizes[a] * sizes[b]
    stats.total_possible_pairs = total
    stats.candidate_pairs = len(pairs)
    return pairs, stats
