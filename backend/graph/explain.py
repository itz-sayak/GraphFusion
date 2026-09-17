"""Evidence-based explanations for every integration decision.

Explanations never just restate a confidence: they list the contributing
signals with their weights and values, the graph adjustments, and the
decisive factor (the signal contributing most to — or most against — the
decision).
"""
from __future__ import annotations

from typing import Any

from backend.core.models import ColumnMatch, DatasetRelationship

SIGNAL_LABELS = {
    "name": ("Name similarity", "name_similarity"),
    "datatype": ("Type compatibility", "datatype_similarity"),
    "semantic": ("Semantic similarity", "semantic_similarity"),
    "value_overlap": ("Value overlap", "value_overlap"),
    "distribution": ("Distribution similarity", "distribution_similarity"),
    "pattern": ("Format pattern similarity", "pattern_similarity"),
    "cardinality": ("Cardinality similarity", "cardinality_similarity"),
}


def explain_column_match(m: ColumnMatch, weights: dict[str, float]) -> dict[str, Any]:
    ev = m.evidence
    total_w = sum(weights.values())
    factors = []
    for key, (label, attr) in SIGNAL_LABELS.items():
        value = getattr(ev, attr)
        w = weights.get(key, 0) / total_w
        factors.append({"signal": key, "label": label, "value": round(value, 4), "weight": round(w, 4), "contribution": round(w * value, 4)})
    factors.sort(key=lambda f: -f["contribution"])
    weakest = min(factors, key=lambda f: f["value"] - 0.5 * f["weight"])
    lines = [f"{m.left.key} ↔ {m.right.key}: confidence {m.score:.2f} ({m.band.value})"]
    for f in factors:
        lines.append(f"  - {f['label']}: {f['value']:.2f} (weight {f['weight']:.2f})")
    lines.append(f"  - Values compared under normaliser '{ev.normalizer}': Jaccard {ev.jaccard:.2f}, containment {ev.containment_left:.2f} → / ← {ev.containment_right:.2f}")
    if abs(ev.graph_adjustment) >= 0.001:
        lines.append(
            f"  - Graph refinement {ev.graph_adjustment:+.3f} (base {ev.base_score:.2f}; two-hop support {ev.transitive_support:.2f}, "
            f"table coherence {ev.structural_support:.2f}, exclusivity {ev.exclusivity:.2f})"
        )
    decision = "accepted" if m.accepted else f"rejected — {m.rejection_reason}"
    lines.append(f"  - Decision: {decision}; relationship type '{m.relationship}'")
    for note in ev.notes:
        if not note.startswith("semantic parts"):
            lines.append(f"  - Note: {note}")
    return {
        "left": m.left.key,
        "right": m.right.key,
        "confidence": m.score,
        "band": m.band.value,
        "accepted": m.accepted,
        "relationship": m.relationship,
        "rejection_reason": m.rejection_reason,
        "factors": factors,
        "strongest_factor": factors[0]["label"],
        "weakest_factor": weakest["label"],
        "normalizer": ev.normalizer,
        "graph": {"base_score": ev.base_score, "adjustment": ev.graph_adjustment, "transitive_support": ev.transitive_support, "structural_support": ev.structural_support, "exclusivity": ev.exclusivity},
        "text": "\n".join(lines),
    }


def explain_relationship(rel: DatasetRelationship, weights: dict[str, float]) -> dict[str, Any]:
    return {
        "datasets": [rel.left_dataset, rel.right_dataset],
        "join_kind": rel.join_kind.value,
        "confidence": rel.confidence,
        "band": rel.band.value,
        "summary": rel.explanation,
        "evidence": rel.evidence,
        "key_matches": [explain_column_match(m, weights) for m in rel.key_matches],
        "attribute_matches": [{"left": m.left.key, "right": m.right.key, "confidence": m.score} for m in rel.attribute_matches],
    }


def explain_route(route: dict[str, Any]) -> str:
    if not route.get("found"):
        return f"No integration route found: {route.get('reason', 'datasets are not connected')}."
    hops = " → ".join(route["path"])
    parts = [f"Best route {hops} (reliability {route['reliability']:.2f}, cost {route['total_cost']:.3f} under {route['cost_mode']} costs)."]
    for h in route["hops"]:
        parts.append(f"  {h['from']} → {h['to']}: {h['join_kind']} at confidence {h['confidence']:.2f} via {', '.join(h['keys']) or 'attribute correspondences'}")
    if route["direct_confidence"] is None:
        parts.append("There is no direct relationship between the two datasets; the route goes through intermediate datasets.")
    elif route["uses_intermediate"]:
        parts.append(f"The direct relationship (confidence {route['direct_confidence']:.2f}) is less reliable than the multi-hop route, so the route is preferred.")
    return "\n".join(parts)
