"""Dataset-level relationship inference.

Aggregates accepted column correspondences between two datasets into one
typed, weighted relationship and decides *how* the two could be combined:

LOOKUP             a column of A references a unique key of B (many-to-one) → left join, A's grain kept
ENTITY_KEY_MERGE   both sides are keyed on the same id space with real overlap → entity fusion on the key
ENTITY_RESOLUTION  same kind of entities but no reliable shared key → probabilistic record linkage
AGGREGATE_LOOKUP   the referenced column is not unique on either side → aggregate B to the key grain first

The confidence of a relationship is interpretable: it is derived from the
confidence of the key correspondence and the fraction of the referencing
rows that actually resolve (row-weighted when the full value distribution is
known), so a perfect name match whose values do not join is *not* a strong
relationship.
"""
from __future__ import annotations

import re
from collections import defaultdict

from backend.core.models import ColumnMatch, ColumnProfile, DataType, DatasetProfile, DatasetRelationship, JoinKind, SemanticType
from backend.core.values import get_normalizer
from backend.matching.scorer import band_for

_MEASURES = {SemanticType.CURRENCY, SemanticType.NUMERIC, SemanticType.LATITUDE, SemanticType.LONGITUDE, SemanticType.FREE_TEXT, SemanticType.BOOLEAN}
_DISCRIMINATIVE = {SemanticType.NAME, SemanticType.EMAIL, SemanticType.PHONE}
_KEYISH = {SemanticType.ID, SemanticType.ZIPCODE, SemanticType.CITY, SemanticType.COUNTRY, SemanticType.CATEGORY, SemanticType.NAME, SemanticType.UNKNOWN}


def row_weighted_coverage(ref: ColumnProfile, target: ColumnProfile, normalizer: str) -> float | None:
    """Fraction of *rows* of ``ref`` whose value exists in ``target`` — exact when both
    value distributions are fully known from profiling (low-cardinality columns)."""
    if not ref.top_values or ref.unique_count > len(ref.top_values) or target.unique_count > len(target.top_values):
        return None
    fn = get_normalizer(normalizer)
    target_vals = {fn(v) for v, _ in target.top_values}
    total = sum(c for _, c in ref.top_values)
    hit = sum(c for v, c in ref.top_values if fn(v) in target_vals)
    return hit / total if total else None


def effective_uniqueness(col: ColumnProfile, profile: DatasetProfile) -> float:
    """Uniqueness net of exact duplicate rows: an export glitch that repeats whole
    rows does not make a primary key semantically non-unique (the merge
    executor deduplicates such rows and reports them)."""
    non_null = profile.row_count - col.null_count
    denom = max(1, non_null - profile.duplicate_rows)
    return min(1.0, col.unique_count / denom) if non_null else 0.0


_KEY_WORDS = {"id", "key", "code", "no", "num", "nr", "fk", "pk", "ref"}


def _name_tokens(name: str) -> set[str]:
    """Content words of a column or table name: camelCase / snake_case / digits split, key words dropped, singular."""
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)
    out = set()
    for w in (w.lower() for w in words):
        if w in _KEY_WORDS or w.isdigit():
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        out.add(w)
    return out


def _tokens_agree(a: set[str], b: set[str]) -> bool:
    return any(x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))) for x in a for y in b)


def chance_overlap(ref: ColumnProfile, tgt: ColumnProfile, observed: float) -> bool:
    """True when the observed containment of ``ref`` in ``tgt`` is what their integer ranges alone produce.

    Auto-increment keys fill 1..n densely, so any two of them overlap: with ``tgt`` dense on [lo, hi], a column
    spread over [rlo, rhi] is expected to be contained to the extent the ranges intersect. An overlap no larger
    than that carries no evidence that the columns refer to the same things."""
    if ref.data_type != DataType.INTEGER or tgt.data_type != DataType.INTEGER:
        return False
    try:
        lo, hi, rlo, rhi = int(tgt.min), int(tgt.max), int(ref.min), int(ref.max)
    except (TypeError, ValueError):
        return False
    span, rspan = hi - lo + 1, rhi - rlo + 1
    if span <= 0 or rspan <= 0 or tgt.unique_count < 0.9 * span:
        return False
    expected = max(0, min(hi, rhi) - max(lo, rlo) + 1) / rspan
    return expected > 0 and observed <= expected + 0.05


def name_support(ref: ColumnProfile, tgt: ColumnProfile, tgt_dataset: str) -> bool:
    """The reference shares a content word with the key it points to (PULocationID → LocationID), or with the
    table that holds it (shipVia → shippers). Generic words (id, key, code, no) do not count."""
    ref_t = _name_tokens(ref.name)
    return _tokens_agree(ref_t, _name_tokens(tgt.name)) or _tokens_agree(ref_t, _name_tokens(tgt_dataset))


def _uninformative(m: ColumnMatch, ca: ColumnProfile, cb: ColumnProfile, da: str, db: str) -> bool:
    """Both directions of an integer correspondence overlap only by range and neither name points to the other."""
    ev = m.evidence
    return (chance_overlap(ca, cb, ev.containment_left) and not name_support(ca, cb, db)) and (
        chance_overlap(cb, ca, ev.containment_right) and not name_support(cb, ca, da))


def _has_key(profile: DatasetProfile, key_unique: float) -> bool:
    """The table identifies its rows with a column of its own (an entity table, not a keyless reference table)."""
    return profile.row_count > 1 and any(
        c.semantic_type not in _MEASURES and c.data_type != DataType.FLOAT and effective_uniqueness(c, profile) >= key_unique for c in profile.columns)


def _aggregatable(profile: DatasetProfile, key_unique: float) -> bool:
    """Summarising the table by a grouping attribute yields meaningful figures: it has no key of its own, or most
    of its columns are quantities."""
    if not _has_key(profile, key_unique):
        return True
    measures = sum(1 for c in profile.columns if c.semantic_type in (SemanticType.NUMERIC, SemanticType.CURRENCY))
    return measures >= 0.5 * len(profile.columns)


def infer_relationships(profiles: dict[str, DatasetProfile], matches: list[ColumnMatch], config: dict) -> list[DatasetRelationship]:
    merge_cfg = config["merge"]
    key_unique = merge_cfg["key_uniqueness"]
    lookup_cov = merge_cfg["lookup_min_containment"]
    entity_overlap = merge_cfg["entity_min_overlap"]
    entity_unique = merge_cfg.get("entity_merge_uniqueness", key_unique)
    thresholds = config["thresholds"]

    by_pair: dict[tuple[str, str], list[ColumnMatch]] = defaultdict(list)
    for m in matches:
        if m.accepted:
            by_pair[(m.left.dataset_id, m.right.dataset_id)].append(m)

    rels: list[DatasetRelationship] = []
    for (da, db), ms in by_pair.items():
        pa, pb = profiles[da], profiles[db]
        candidates: list[tuple[JoinKind, float, list[ColumnMatch], dict]] = []

        lookups_by_dim: dict[str, list[tuple[ColumnMatch, float, str]]] = defaultdict(list)
        for m in ms:
            ca, cb = pa.column(m.left.column), pb.column(m.right.column)
            # containment into a unique key already checked by the aligner; the exception to the measure veto is only for
            # integer codes stored as numbers (zero-padded account numbers), never money, text, flags, coordinates or floats
            verified_fk = m.relationship == "foreign_key_candidate" and all(
                c.semantic_type not in _MEASURES or (c.semantic_type == SemanticType.NUMERIC and c.data_type == DataType.INTEGER) for c in (ca, cb))
            if ca.semantic_type in _MEASURES and cb.semantic_type in _MEASURES and not verified_fk:
                continue
            ev = m.evidence
            # a reference into a unique key (either direction)
            for ref, tgt, cont, dim in ((ca, cb, ev.containment_left, db), (cb, ca, ev.containment_right, da)):
                tgt_profile = pb if tgt is cb else pa
                if chance_overlap(ref, tgt, cont) and not name_support(ref, tgt, dim):
                    continue  # two surrogate-key ranges overlap by construction; without name evidence it is no link
                if effective_uniqueness(tgt, tgt_profile) >= key_unique and tgt.unique_count > 1 and (ref.semantic_type in _KEYISH or verified_fk):
                    coverage = row_weighted_coverage(ref, tgt, ev.normalizer)
                    cov = coverage if coverage is not None else cont
                    if cov >= lookup_cov:
                        lookups_by_dim[dim].append((m, cov, "row-weighted" if coverage is not None else "distinct"))

        if len(lookups_by_dim) == 2:
            # both sides look like a key: the referenced side is the *more* unique one (a near-unique reference
            # column with a few repeats must not become the dimension of its own parent)
            def key_uniq(dim: str) -> float:
                prof = pb if dim == db else pa
                return max(effective_uniqueness(prof.column(m.right.column if dim == db else m.left.column), prof) for m, _, _ in lookups_by_dim[dim])
            ua, ub = key_uniq(da), key_uniq(db)
            if abs(ua - ub) > 1e-9:
                weaker = da if ua < ub else db
                del lookups_by_dim[weaker]
        for dim, items in lookups_by_dim.items():
            # several fact columns referencing the *same* key column are roles (pickup / dropoff → LocationID);
            # references to *different* columns of the dimension (customerID, address, companyName) describe one
            # link: the best identifier is the key, the others are matching attributes
            by_key: dict[str, list] = defaultdict(list)
            for it in items:
                by_key[it[0].right.column if dim == db else it[0].left.column].append(it)
            if len(by_key) > 1:
                dprof = pb if dim == db else pa

                def key_rank(col: str) -> tuple:
                    c = dprof.column(col)
                    return (c.semantic_type == SemanticType.ID, max(it[0].score for it in by_key[col]), effective_uniqueness(c, dprof))
                items = by_key[max(by_key, key=key_rank)]
            both_unique = all(
                effective_uniqueness(pa.column(m.left.column), pa) >= entity_unique and effective_uniqueness(pb.column(m.right.column), pb) >= entity_unique
                for m, _, _ in items
            )
            best = max(items, key=lambda it: it[0].score)
            m, cov, how = best
            jacc = m.evidence.jaccard
            if both_unique and jacc >= entity_overlap and pa.row_count and pb.row_count:
                conf = 0.7 * m.score + 0.3 * jacc
                candidates.append((JoinKind.ENTITY_KEY_MERGE, conf, [m], {"jaccard": jacc, "coverage": cov}))
            else:
                conf = 0.7 * m.score + 0.3 * cov
                fact = da if dim == db else db
                candidates.append((JoinKind.LOOKUP, conf, [it[0] for it in items], {"dimension": dim, "fact": fact, "coverage": round(cov, 4), "coverage_basis": how, "references": len(items)}))

        # probabilistic entity resolution: descriptive, discriminative correspondences
        descriptive = [m for m in ms if pa.column(m.left.column).semantic_type not in _MEASURES]
        discriminative = [m for m in descriptive if pa.column(m.left.column).semantic_type in _DISCRIMINATIVE or pb.column(m.right.column).semantic_type in _DISCRIMINATIVE]
        # two tables describe common entities only if some identifying value (name, e-mail, phone) actually recurs
        shared = max((max(m.evidence.containment_left, m.evidence.containment_right) for m in discriminative), default=0.0)
        if len(descriptive) >= 2 and discriminative and shared >= float(merge_cfg.get("entity_min_shared_values", 0.02)):
            top = sorted((m.score for m in descriptive), reverse=True)[:3]
            both_entity_like = min(pa.row_count, pb.row_count) > 0 and max(pa.row_count, pb.row_count) / max(1, min(pa.row_count, pb.row_count)) < 20
            if both_entity_like:
                conf = (sum(top) / len(top)) * (0.95 if len(discriminative) >= 2 else 0.85)
                candidates.append((JoinKind.ENTITY_RESOLUTION, conf, discriminative, {"descriptive_matches": len(descriptive), "discriminative_matches": len(discriminative)}))

        # aggregate lookup: shared low-cardinality categorical key
        if not lookups_by_dim:
            for m in ms:
                ca, cb = pa.column(m.left.column), pb.column(m.right.column)
                if ca.semantic_type in _MEASURES or cb.semantic_type in _MEASURES or _uninformative(m, ca, cb, da, db):
                    continue
                # an attribute unique on neither side (state, country, job title) joins two tables only if summarising
                # one of them by it means something: a keyless reference table (geolocation points per zip) or a
                # table of quantities (demographics per borough). Two tables of described entities merely share it.
                shared_attribute = max(effective_uniqueness(ca, pa), effective_uniqueness(cb, pb)) < key_unique
                if shared_attribute and not (_aggregatable(pa, key_unique) or _aggregatable(pb, key_unique)):
                    continue
                cov_ab = row_weighted_coverage(ca, cb, m.evidence.normalizer)
                cov_ba = row_weighted_coverage(cb, ca, m.evidence.normalizer)
                cov = max(c for c in (cov_ab, cov_ba, m.evidence.containment_left, m.evidence.containment_right) if c is not None)
                if cov >= lookup_cov and m.score >= thresholds["medium"] * 0.9:
                    ref_is_a = pa.row_count >= pb.row_count
                    conf = 0.85 * (0.7 * m.score + 0.3 * cov)
                    candidates.append((JoinKind.AGGREGATE_LOOKUP, conf, [m], {"fact": da if ref_is_a else db, "dimension": db if ref_is_a else da, "coverage": round(cov, 4)}))

        if not candidates:
            continue
        priority = {JoinKind.ENTITY_KEY_MERGE: 3, JoinKind.LOOKUP: 2, JoinKind.ENTITY_RESOLUTION: 1, JoinKind.AGGREGATE_LOOKUP: 0}
        # most reliable relationship wins; structural priority only breaks near-ties (within 0.05)
        top = max(c[1] for c in candidates)
        kind, conf, keys, info = max((c for c in candidates if c[1] >= top - 0.05), key=lambda c: (priority[c[0]], c[1]))
        conf = round(min(1.0, conf), 4)
        attrs = [m for m in ms if m not in keys]
        rel = DatasetRelationship(
            left_dataset=da,
            right_dataset=db,
            confidence=conf,
            band=band_for(conf, thresholds),
            join_kind=kind,
            key_matches=keys,
            attribute_matches=attrs,
            evidence={**info, "alternatives": [{"join_kind": c[0].value, "confidence": round(c[1], 4)} for c in candidates if c[0] != kind]},
        )
        rel.explanation = explain_relationship(rel)
        rels.append(rel)
    rels = drop_implied_sibling_links(rels, profiles, entity_unique)
    rels.sort(key=lambda r: -r.confidence)
    return rels


def drop_implied_sibling_links(rels: list[DatasetRelationship], profiles: dict[str, DatasetProfile], strict_unique: float) -> list[DatasetRelationship]:
    """Remove direct links between *sibling* tables (fan-trap avoidance).

    Two child tables that both reference the same strictly unique key of a parent
    (payments.order_id → orders.order_id ← reviews.order_id) share values, so they also
    look related to each other. Joining them directly multiplies rows (every payment ×
    every review of an order); the correct path goes through the parent. A lookup or
    aggregated lookup X.x — Y.y where at least one side is not strictly unique is dropped
    when X.x and Y.y both reference one and the same strictly unique key Z.z. Its column
    matches are un-accepted with the reason, so they appear as candidates only.
    """
    def strict(ds: str, col: str) -> bool:
        p = profiles[ds]
        return effective_uniqueness(p.column(col), p) >= strict_unique

    refs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)  # column → strictly unique keys it references
    for r in rels:
        if r.join_kind not in (JoinKind.LOOKUP, JoinKind.ENTITY_KEY_MERGE):
            continue
        for m in r.key_matches:
            a, b = (m.left.dataset_id, m.left.column), (m.right.dataset_id, m.right.column)
            if strict(*b):
                refs[a].add(b)
            if strict(*a):
                refs[b].add(a)
    kept = []
    for r in rels:
        if r.join_kind in (JoinKind.LOOKUP, JoinKind.AGGREGATE_LOOKUP) and r.key_matches:
            m = max(r.key_matches, key=lambda x: x.score)
            a, b = (m.left.dataset_id, m.left.column), (m.right.dataset_id, m.right.column)
            if not (strict(*a) and strict(*b)):
                parents = sorted((refs[a] & refs[b]) - {a, b})
                if parents:
                    via = f"{parents[0][0]}.{parents[0][1]}"
                    reason = f"sibling tables: both columns reference {via}; joining them directly would multiply rows, so they are connected through {parents[0][0]}"
                    for x in r.key_matches + r.attribute_matches:
                        x.accepted = False
                        x.rejection_reason = reason
                    continue
        kept.append(r)
    return kept


def explain_relationship(rel: DatasetRelationship) -> str:
    keys = ", ".join(f"{m.left.column} ↔ {m.right.column} ({m.score:.2f}, values via '{m.evidence.normalizer}')" for m in rel.key_matches)
    ev = rel.evidence
    if rel.join_kind == JoinKind.LOOKUP:
        return (
            f"{ev['fact']} references {ev['dimension']} through {keys}. "
            f"{ev['coverage']:.0%} of referencing values ({ev['coverage_basis']}) resolve to a unique key, "
            f"so rows of {ev['fact']} can be enriched without changing their grain."
        )
    if rel.join_kind == JoinKind.ENTITY_KEY_MERGE:
        return f"Both datasets are keyed on the same identifier space ({keys}); key overlap Jaccard = {ev['jaccard']:.2f}. Records with equal keys describe the same entity."
    if rel.join_kind == JoinKind.ENTITY_RESOLUTION:
        return (
            f"No shared key, but {ev['descriptive_matches']} descriptive attributes correspond, including discriminative ones ({keys}). "
            "Records are linked probabilistically (Fellegi–Sunter) rather than joined."
        )
    return f"Only a non-unique categorical key connects them ({keys}, coverage {ev['coverage']:.0%}); {ev['dimension']} must be aggregated to that grain before joining."
