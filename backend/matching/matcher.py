"""Schema-matching pipeline.

    candidates (blocking) → weighted scoring → similarity flooding → bipartite alignment

The ``strategy`` knobs exist so the evaluation harness can run every ablation
variant through this same code path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.core.models import ColumnMatch, ColumnRef, DatasetProfile, SemanticType
import numpy as np

from backend.config import PROJECT_ROOT
from backend.logging_conf import get_logger
from backend.matching.learned import LearnedMatcher, load_default, pair_features
from backend.matching.aligner import align
from backend.matching.candidates import candidate_pairs
from backend.matching.embeddings import EmbeddingIndex, make_encoder
from backend.matching.scorer import ALL_SIGNALS, ColumnContext, band_for, score_pair
from backend.matching.similarity_flooding import flood
from backend.profiling.sketches import ColumnSketch

log = get_logger(__name__)


@dataclass
class MatchStrategy:
    signals: tuple[str, ...] = ALL_SIGNALS
    flooding: bool = True
    bipartite: bool = True
    name: str = "full_graph"
    scorer: str | None = None  # None → config schema_matching.scorer


@dataclass
class MatchingResult:
    matches: list[ColumnMatch]
    all_scores: dict[tuple[str, str], float]
    base_scores: dict[tuple[str, str], float]
    candidate_strategy: str
    candidate_pairs: int
    flooding: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    evidence: dict = field(default_factory=dict)  # (left key, right key) -> MatchEvidence, for every scored pair
    columns: dict = field(default_factory=dict)  # column key -> ColumnProfile
    scorer: str = "weighted"

    def accepted(self) -> list[ColumnMatch]:
        return [m for m in self.matches if m.accepted]


def build_contexts(profiles: dict[str, DatasetProfile], sketches: dict[str, dict[str, ColumnSketch]]) -> list[ColumnContext]:
    out: list[ColumnContext] = []
    for ds_id, prof in profiles.items():
        for col in prof.columns:
            out.append(ColumnContext(ds_id, col, sketches[ds_id][col.name], prof.row_count))
    return out


def match_schemas(
    profiles: dict[str, DatasetProfile],
    sketches: dict[str, dict[str, ColumnSketch]],
    config: dict,
    strategy: MatchStrategy | None = None,
    min_accept: float | None = None,
    learned: "LearnedMatcher | None" = None,
    keep_pairs: set[tuple[str, str]] | None = None,
    crosswalks: dict[frozenset[str], str] | None = None,
) -> MatchingResult:
    started = time.perf_counter()
    strategy = strategy or MatchStrategy()
    sm = config["schema_matching"]
    thresholds = config["thresholds"]
    vendor = tuple(config["merge"].get("vendor_prefixes", []))
    contexts = build_contexts(profiles, sketches)
    ctx = {c.key: c for c in contexts}

    pairs, cand_strategy = candidate_pairs(contexts, config)
    crosswalks = crosswalks or {}
    if crosswalks:
        # column pairs with a verified value crosswalk are always scored, with the crosswalk as a candidate normaliser
        known = {frozenset(p) for p in pairs}
        pairs = list(pairs) + [tuple(sorted(p)) for p in crosswalks if p not in known and all(k in ctx for k in p)]

    index = None
    if "semantic" in strategy.signals:
        index = EmbeddingIndex(make_encoder(config)).build({c.key: c.profile for c in contexts}, vendor)

    base: dict[tuple[str, str], float] = {}
    evidence = {}
    for ka, kb in pairs:
        a, b = ctx[ka], ctx[kb]
        if (a.dataset_id, a.profile.name) > (b.dataset_id, b.profile.name):
            a, b = b, a
        extra = (crosswalks[frozenset((a.key, b.key))],) if frozenset((a.key, b.key)) in crosswalks else ()
        score, ev = score_pair(a, b, sm["weights"], strategy.signals, index, vendor, extra)
        key = (a.key, b.key)
        base[key] = score
        evidence[key] = ev

    dataset_of = {c.key: c.dataset_id for c in contexts}
    flood_info: dict = {}
    refined = dict(base)
    if strategy.flooding and sm["flooding"]["enabled"]:
        refined, flood_info = flood(base, dataset_of, evidence, sm["flooding"])

    scorer = strategy.scorer or sm.get("scorer", "weighted")
    if scorer in ("learned", "blend") and refined:
        model = learned or load_default(PROJECT_ROOT / sm.get("learned_model", "models/column_matcher.joblib"))
        if model is None:
            log.warning("Learned matcher requested but no model found; using weighted scores", path=sm.get("learned_model"))
            scorer = "weighted"
        else:
            keys = list(refined)
            X = np.array([pair_features(evidence[k], ctx[k[0]].profile, ctx[k[1]].profile) for k in keys])
            probs = model.predict(X)
            for k, p in zip(keys, probs):
                evidence[k].learned_probability = round(float(p), 4)
                refined[k] = round(float(p) if scorer == "learned" else 0.5 * (refined[k] + float(p)), 4)

    matches: list[ColumnMatch] = []
    for (ka, kb), score in refined.items():
        if score < sm["min_candidate_score"] and tuple(sorted((ka, kb))) not in (keep_pairs or set()):
            continue  # user-decided pairs stay visible whatever their score
        a, b = ctx[ka], ctx[kb]
        matches.append(
            ColumnMatch(
                left=ColumnRef(dataset_id=a.dataset_id, column=a.profile.name),
                right=ColumnRef(dataset_id=b.dataset_id, column=b.profile.name),
                score=score,
                band=band_for(score, thresholds),
                evidence=evidence[(ka, kb)],
            )
        )

    uniqueness = {c.key: c.profile.uniqueness for c in contexts}
    # identifiers are defined by their values: two identifier columns that share (almost) no values are different
    # identifier systems, however similar their names, types and formats look
    id_min_overlap = float(sm["alignment"].get("identifier_min_overlap", 0.05))
    vetoed: dict[tuple[str, str], str] = {}
    for m in matches:
        pa, pb = ctx[m.left.key].profile, ctx[m.right.key].profile
        if pa.semantic_type == SemanticType.ID and pb.semantic_type == SemanticType.ID:
            overlap = max(m.evidence.containment_left, m.evidence.containment_right)
            if overlap < id_min_overlap and tuple(sorted(m.pair_key)) not in (keep_pairs or set()):
                vetoed[m.pair_key] = f"identifier columns share almost no values (containment {overlap:.2f} < {id_min_overlap:.2f})"
    accept_at = min_accept if min_accept is not None else config["merge_modes"][config["merge"]["default_mode"]]["min_edge"]
    align(
        matches,
        min_score=accept_at,
        key_uniqueness=config["merge"]["key_uniqueness"],
        fk_min_containment=sm["alignment"]["fk_min_containment"],
        uniqueness=uniqueness,
        allow_fk=sm["alignment"]["allow_fk_many_to_one"],
        one_to_one=strategy.bipartite and sm["alignment"]["one_to_one"],
        vetoed=vetoed,
    )
    matches.sort(key=lambda m: (-m.accepted, -m.score))
    elapsed = (time.perf_counter() - started) * 1000
    log.info(
        "Candidate relationships discovered",
        strategy=strategy.name,
        candidate_pairs=len(pairs),
        candidates=cand_strategy,
        scored_above_min=len(matches),
        accepted=sum(m.accepted for m in matches),
        flooding_iterations=flood_info.get("iterations"),
        elapsed_ms=round(elapsed, 1),
    )
    return MatchingResult(matches, refined, base, cand_strategy, len(pairs), flood_info, round(elapsed, 1),
                          evidence=evidence, columns={c.key: c.profile for c in contexts}, scorer=scorer)
