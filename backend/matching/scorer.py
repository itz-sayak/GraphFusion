"""Weighted column-compatibility score S(ci, cj).

    S = Σ_k w_k · s_k(ci, cj) / Σ_k w_k        over the enabled signals k

Signals: name (N), datatype (T), semantic (E), value_overlap (V),
distribution (D), pattern (P), cardinality (C). Weights come from
configuration and are renormalised over whichever subset is enabled, which is
how the ablation study switches signals on and off without touching code.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.core.models import ColumnProfile, ConfidenceBand, MatchEvidence
from backend.matching.dist_sim import cardinality_similarity, distribution_similarity, pattern_similarity
from backend.matching.embeddings import EmbeddingIndex, semantic_similarity
from backend.matching.name_sim import name_similarity
from backend.matching.type_sim import datatype_similarity, semantic_type_compatibility
from backend.matching.value_sim import value_similarity
from backend.profiling.sketches import ColumnSketch

ALL_SIGNALS = ("name", "datatype", "semantic", "value_overlap", "distribution", "pattern", "cardinality")


@dataclass
class ColumnContext:
    dataset_id: str
    profile: ColumnProfile
    sketch: ColumnSketch
    row_count: int

    @property
    def key(self) -> str:
        return f"{self.dataset_id}::{self.profile.name}"


def band_for(score: float, thresholds: dict[str, float]) -> ConfidenceBand:
    if score >= thresholds["high"]:
        return ConfidenceBand.HIGH
    if score >= thresholds["medium"]:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def score_pair(
    a: ColumnContext,
    b: ColumnContext,
    weights: dict[str, float],
    signals: tuple[str, ...] = ALL_SIGNALS,
    index: EmbeddingIndex | None = None,
    vendor_prefixes: tuple[str, ...] = (),
    extra_normalizers: tuple[str, ...] = (),
) -> tuple[float, MatchEvidence]:
    pa, pb = a.profile, b.profile
    ev = MatchEvidence()
    values: dict[str, float] = {}
    if "name" in signals:
        values["name"] = ev.name_similarity = name_similarity(pa.name, pb.name, vendor_prefixes)
    if "datatype" in signals:
        values["datatype"] = ev.datatype_similarity = datatype_similarity(pa.data_type, pb.data_type)
    if "semantic" in signals:
        sem, parts = semantic_similarity(pa, pb, a.key, b.key, index, vendor_prefixes)
        values["semantic"] = ev.semantic_similarity = sem
        ev.notes.append(f"semantic parts: concept={parts['concept']}, semantic_type={parts['semantic_type']}, embedding={parts['embedding']}")
    if "value_overlap" in signals:
        vs = value_similarity(pa, a.sketch, pb, b.sketch, extra_normalizers)
        values["value_overlap"] = ev.value_overlap = float(vs["value_overlap"])
        ev.jaccard = float(vs["jaccard"])
        ev.containment_left = float(vs["containment_left"])
        ev.containment_right = float(vs["containment_right"])
        ev.normalizer = str(vs["normalizer"])
        if vs.get("note"):
            ev.notes.append(str(vs["note"]))
    if "distribution" in signals:
        values["distribution"] = ev.distribution_similarity = distribution_similarity(pa, pb)
    if "pattern" in signals:
        values["pattern"] = ev.pattern_similarity = pattern_similarity(pa, pb)
        if ev.normalizer.startswith("crosswalk:"):
            # a verified crosswalk maps one vocabulary onto the other ("DEU" -> "germany"): their surface formats differ by
            # construction, so format agreement is measured where the values were compared, i.e. after the mapping
            mapped = max(ev.containment_left, ev.containment_right)
            if mapped > ev.pattern_similarity:
                values["pattern"] = ev.pattern_similarity = round(mapped, 4)
                ev.notes.append("pattern similarity measured after the verified value crosswalk")
    if "cardinality" in signals:
        values["cardinality"] = ev.cardinality_similarity = cardinality_similarity(pa, pb)

    total_w = sum(weights[k] for k in values)
    score = sum(weights[k] * v for k, v in values.items()) / total_w if total_w else 0.0

    if "semantic" in signals and semantic_type_compatibility(pa.semantic_type, pb.semantic_type) == 0.0 and min(pa.semantic_confidence, pb.semantic_confidence) >= 0.6:
        score *= 0.6
        ev.notes.append(f"semantic type conflict ({pa.semantic_type.value} vs {pb.semantic_type.value}) → ×0.6")
    ev.base_score = round(score, 4)
    return round(score, 4), ev
