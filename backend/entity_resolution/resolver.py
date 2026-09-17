"""Entity resolution orchestration.

    load fields → blocking → comparison vectors → u from random pairs →
    EM for m/λ → match probabilities → correlation clustering → entity ids

Works for deduplication of one dataset, linkage across datasets, or both at
once (records of several datasets resolved jointly).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

import networkx as nx

from backend.core.models import EntityCluster, EntityPair, SemanticType
from backend.entity_resolution.blocking import BlockingStats, generate_candidates
from backend.entity_resolution.clustering import connected_components_clusters, correlation_clusters
from backend.entity_resolution.comparators import comparator_for
from backend.entity_resolution.fellegi_sunter import (
    FSModel, compare_pair, estimate_u, fit_label_calibration, fit_rule_blocked_em, prior_over_all_pairs, term_frequencies,
)
from backend.logging_conf import get_logger
from backend.storage.duck import connect, parquet_scan, quote_ident

log = get_logger(__name__)


@dataclass
class FieldSpec:
    """A canonical attribute compared across sources."""

    name: str
    semantic_type: SemanticType
    columns: dict[str, str]  # dataset_id -> source column
    normalizer: str = "basic"


@dataclass
class ERResult:
    pairs: list[EntityPair]
    clusters: list[EntityCluster]
    assignment: dict[tuple[str, int], str]  # (dataset_id, row) -> entity_id
    model: FSModel
    blocking: BlockingStats
    stats: dict[str, Any] = field(default_factory=dict)
    pair_probabilities: dict[tuple[tuple[str, int], tuple[str, int]], float] = field(default_factory=dict)  # every compared pair
    labels_applied: int = 0

    def cluster_of(self, dataset_id: str, row: int) -> str | None:
        return self.assignment.get((dataset_id, row))


def load_records(storage_paths: dict[str, str], fields: list[FieldSpec]) -> dict[tuple[int, int], dict[str, Any]]:
    """Read only the compared columns (projection pushdown) with a stable row index."""
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for di, (ds, path) in enumerate(storage_paths.items()):
        cols = [(f.name, f.columns[ds]) for f in fields if ds in f.columns]
        if not cols:
            continue
        select = ", ".join(f"{quote_ident(src)} AS {quote_ident(name)}" for name, src in cols)
        with connect() as con:
            src = f"read_parquet('{Path(path).as_posix()}', file_row_number=true)"
            cur = con.execute(f"SELECT file_row_number AS __row, {select} FROM {src}")
            names = [d[0] for d in cur.description]
            for row in cur.fetchall():
                rec = dict(zip(names, row))
                records[(di, int(rec.pop("__row")))] = rec
    return records


def resolve(
    storage_paths: dict[str, str],
    fields: list[FieldSpec],
    config: dict,
    dedupe: set[str] | None = None,
    auto_threshold: float = 0.9,
    review_threshold: float = 0.7,
    clustering: str | None = None,
    entity_prefix: str = "ENT",
    records: dict[tuple[int, int], dict[str, Any]] | None = None,
    labels: dict[tuple[tuple[str, int], tuple[str, int]], bool] | None = None,
    calibration_pairs: set[tuple[tuple[str, int], tuple[str, int]]] | None = None,
) -> ERResult:
    """``labels`` become must-/cannot-link constraints. ``calibration_pairs`` marks the labelled pairs
    that were sampled uniformly at random (an unbiased sample); with ``label_calibration: random_only``
    only those recalibrate the match weights, because uncertainty-sampled labels are biased."""
    started = time.perf_counter()
    er_cfg = config["entity_resolution"]
    datasets = list(storage_paths)
    dedupe = dedupe if dedupe is not None else set(datasets)
    records = records if records is not None else load_records(storage_paths, fields)
    comparators = {}
    comparator_names = {}
    for f in fields:
        cname, fn = comparator_for(f.semantic_type, f.normalizer)
        comparators[f.name] = fn
        comparator_names[f.name] = cname

    candidates, bstats = generate_candidates(
        records,
        [(f.name, f.semantic_type, f.normalizer) for f in fields],
        allow_within={i for i, d in enumerate(datasets) if d in dedupe},
        max_block_size=er_cfg["max_block_size"],
        window=er_cfg["sorted_neighborhood_window"],
        schemes=er_cfg["blocking"],
        refine_block_size=er_cfg.get("refine_block_size"),
    )
    t_block = time.perf_counter()

    # user labels (active learning): labelled pairs are always compared, even if blocking missed them
    ds_index = {d: i for i, d in enumerate(datasets)}
    constraints: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}
    calib_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (ra, rb), is_match in (labels or {}).items():
        if ra[0] not in ds_index or rb[0] not in ds_index:
            continue
        a, b = (ds_index[ra[0]], int(ra[1])), (ds_index[rb[0]], int(rb[1]))
        if a == b or a not in records or b not in records:
            continue
        pair = (a, b) if a < b else (b, a)
        constraints[pair] = bool(is_match)
        candidates.add(pair)
        if calibration_pairs and ((ra, rb) in calibration_pairs or (rb, ra) in calibration_pairs):
            calib_set.add(pair)
    cand_list = sorted(candidates)
    levels_list, scores_list = [], []
    for a, b in cand_list:
        lv, sc = compare_pair(records[a], records[b], comparators)
        levels_list.append(lv)
        scores_list.append(sc)

    u = estimate_u(list(records.values()), comparators)
    n_rec = len(records)
    prior = min(0.5, max(er_cfg["prior_match_rate"], (n_rec / 2) / max(1, len(cand_list))))
    rule_fields = [f.name for f in fields if f.semantic_type in (SemanticType.EMAIL, SemanticType.PHONE, SemanticType.ID, SemanticType.NAME)]
    labelled_matches = [lv for pair, lv in zip(cand_list, levels_list) if constraints.get(pair) is True]
    model = fit_rule_blocked_em(levels_list, u, [f.name for f in fields], rule_fields, prior=prior, iterations=er_cfg["em_iterations"],
                                labelled_matches=labelled_matches, label_weight=er_cfg.get("label_weight", 5.0))

    # ---- calibration of the unsupervised posterior
    # term frequencies: agreeing on a common value ("Rahul Sharma") is weaker evidence than agreeing on a rare one
    tf_fields: dict[str, tuple[dict[Any, float], Any]] = {}
    if er_cfg.get("term_frequency", False):
        from backend.core.values import basic_normalize, digits_normalize, get_normalizer
        from backend.entity_resolution.comparators import normalize_person_name

        keys = {SemanticType.NAME: normalize_person_name, SemanticType.EMAIL: basic_normalize, SemanticType.PHONE: digits_normalize}
        recs = list(records.values())
        for f in fields:
            key = keys.get(f.semantic_type) or (get_normalizer(f.normalizer) if f.semantic_type in (SemanticType.ID, SemanticType.ZIPCODE) else None)
            if key is None:
                continue
            freq = term_frequencies(recs, f.name, key)
            if freq:
                tf_fields[f.name] = (freq, key)

    def u_exact_for(a: tuple[int, int], lv: dict[str, str]) -> dict[str, float] | None:
        out = {}
        for fname, (freq, key) in tf_fields.items():
            if lv.get(fname) == "exact":
                v = records[a].get(fname)
                k = key(v) if v is not None else None
                if k in freq:
                    out[fname] = freq[k]
        return out or None

    u_overrides = [u_exact_for(a, lv) for (a, _), lv in zip(cand_list, levels_list)] if tf_fields else [None] * len(cand_list)
    if er_cfg.get("prior_population", "candidates") == "all_pairs":
        # the prior must describe the same population as u (all pairs), not the blocked candidates
        llrs = [model.llr(lv, uo) for lv, uo in zip(levels_list, u_overrides)]
        total = bstats.total_possible_pairs
        model.prior = prior_over_all_pairs(llrs, total, start=model.prior * len(cand_list) / max(1, total))

    # label-driven recalibration: every label informs all pairs with a similar match weight
    calibration = None
    mode = er_cfg.get("label_calibration", "random_only")
    mode = "all" if mode is True else ("off" if mode is False else mode)
    calib = constraints if mode == "all" else ({p: constraints[p] for p in calib_set} if mode == "random_only" else {})
    if calib:
        lab_w = [model.probability(lv, uo)[1] for pair, lv, uo in zip(cand_list, levels_list, u_overrides) if pair in calib]
        lab_y = [calib[pair] for pair in cand_list if pair in calib]
        all_w = [model.probability(lv, uo)[1] for lv, uo in zip(levels_list, u_overrides)]
        calibration = fit_label_calibration(lab_w, lab_y, all_w, label_weight=er_cfg.get("calibration_label_weight", 20.0))

    pairs: list[EntityPair] = []
    edges: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for (a, b), lv, sc, uo in zip(cand_list, levels_list, scores_list, u_overrides):
        p, w = model.probability(lv, uo)
        if calibration is not None:
            z = max(-40.0, min(40.0, calibration[0] * w + calibration[1]))
            p = 1.0 / (1.0 + math.exp(-z))
        if (a, b) in constraints:
            p = 1.0 if constraints[(a, b)] else 0.0
        edges[(a, b)] = p
        if p >= review_threshold * 0.5:
            pairs.append(
                EntityPair(
                    left_dataset=datasets[a[0]], left_row=a[1], right_dataset=datasets[b[0]], right_row=b[1],
                    probability=round(p, 4), match_weight=round(w, 3), field_levels=lv, field_scores=sc,
                )
            )
    t_score = time.perf_counter()

    method = clustering or er_cfg["clustering"]
    nodes = list(records)
    # only links at or above the auto-merge threshold may create entities; the
    # review band is surfaced to the user but never merged automatically
    if method == "connected_components":
        groups = connected_components_clusters(nodes, edges, auto_threshold, constraints)
    else:
        groups = correlation_clusters(nodes, edges, threshold=auto_threshold, constraints=constraints)

    groups.sort(key=lambda g: min(g))
    assignment: dict[tuple[str, int], str] = {}
    clusters: list[EntityCluster] = []
    for i, g in enumerate(groups):
        eid = f"{entity_prefix}{i:07d}"
        internal = [edges[(a, b)] for a in g for b in g if a < b and (a, b) in edges]
        conf = sum(internal) / len(internal) if internal else 1.0
        members = sorted((datasets[d], r) for d, r in g)
        for m in members:
            assignment[m] = eid
        clusters.append(EntityCluster(entity_id=eid, members=members, confidence=round(conf, 4), size=len(members)))

    review = sum(1 for p in edges.values() if review_threshold <= p < auto_threshold)
    stats = {
        "records": n_rec,
        "candidate_pairs": len(cand_list),
        "possible_pairs": bstats.total_possible_pairs,
        "reduction_ratio": round(bstats.reduction_ratio, 6),
        "blocking_schemes": bstats.schemes,
        "skipped_blocks": bstats.skipped_blocks,
        "refined_blocks": bstats.refined_blocks,
        "auto_links": sum(1 for p in edges.values() if p >= auto_threshold),
        "review_links": review,
        "entities": len(clusters),
        "multi_record_entities": sum(1 for c in clusters if c.size > 1),
        "clustering": method,
        "comparators": comparator_names,
        "timings_ms": {
            "blocking": round((t_block - started) * 1000, 1),
            "scoring": round((t_score - t_block) * 1000, 1),
            "clustering": round((time.perf_counter() - t_score) * 1000, 1),
        },
    }
    log.info("Entity resolution completed", **{k: v for k, v in stats.items() if k in ("records", "candidate_pairs", "auto_links", "review_links", "entities", "clustering")})
    stats["user_labels"] = len(constraints)
    stats["prior"] = {"population": er_cfg.get("prior_population", "candidates"), "lambda": model.prior, "term_frequency": bool(tf_fields)}
    stats["label_calibration"] = None if calibration is None else {"a": round(calibration[0], 4), "b": round(calibration[1], 4)}
    return ERResult(pairs=sorted(pairs, key=lambda p: -p.probability), clusters=clusters, assignment=assignment, model=model, blocking=bstats, stats=stats,
                    pair_probabilities={((datasets[a[0]], a[1]), (datasets[b[0]], b[1])): p for (a, b), p in edges.items()},
                    labels_applied=len(constraints))


def review_queue(result: ERResult, limit: int = 20, threshold: float = 0.9, low: float = 0.01, high: float = 0.999, exclude: set | None = None) -> list[dict[str, Any]]:
    """Margin (uncertainty) sampling around the merge decision boundary.

    Merges happen at p ≥ ``threshold``, not at 0.5, so uncertainty is measured in
    log-odds distance from the boundary:  u = exp(−|logit(p) − logit(t)|) ∈ (0, 1].
    Pairs where the clustering outcome disagrees with the pairwise decision
    (merged although p < t, or separated although p ≥ t) get +0.5.
    Pairs already labelled are excluded.
    """
    exclude = exclude or set()
    lt = math.log(threshold / (1 - threshold))
    items = []
    for (a, b), p in result.pair_probabilities.items():
        if not (low < p < high) or (a, b) in exclude or (b, a) in exclude:
            continue
        u = math.exp(-abs(math.log(p / (1 - p)) - lt))
        same = result.assignment.get(a) is not None and result.assignment.get(a) == result.assignment.get(b)
        disagreement = (same and p < threshold) or (not same and p >= threshold)
        items.append({"left": {"dataset": a[0], "row": a[1]}, "right": {"dataset": b[0], "row": b[1]}, "probability": round(p, 4),
                      "margin_uncertainty": round(u, 4), "same_entity_now": same, "disagreement": disagreement, "score": round(u + (0.5 if disagreement else 0.0), 4)})
    items.sort(key=lambda x: -x["score"])
    return items[:limit]


def one_to_one_links(pairs: list[EntityPair], threshold: float) -> list[EntityPair]:
    """Optional 1:1 linkage between two deduplicated sources (maximum-weight bipartite matching)."""
    g = nx.Graph()
    for p in pairs:
        if p.probability >= threshold and p.left_dataset != p.right_dataset:
            g.add_edge((p.left_dataset, p.left_row), (p.right_dataset, p.right_row), weight=p.probability, pair=p)
    matching = nx.max_weight_matching(g)
    return [g.edges[a, b]["pair"] for a, b in matching]
