"""The integration graph G = (V, E).

V = datasets ∪ columns ∪ entities
E = contains (dataset → column)
  ∪ similar_column / foreign_key_candidate / value_overlap / semantic_match (column ↔ column)
  ∪ dataset relationships (dataset ↔ dataset, typed by join kind)
  ∪ same_entity / derived_from (added after entity resolution and merging)

Every edge carries ``weight`` (= confidence), ``confidence``, ``cost`` (for
shortest-path reasoning), ``relationship_type`` and an ``evidence`` payload
so any path through the graph can be explained edge by edge.
"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

from backend.core.models import ColumnMatch, DatasetArtifact, DatasetProfile, DatasetRelationship


def edge_cost(confidence: float, mode: str = "neglog") -> float:
    """cost(e) = −log c(e)  (max-reliability paths)  or  1 − c(e)  (linear, original spec)."""
    c = min(max(confidence, 1e-9), 1.0)
    if mode == "linear":
        return 1.0 - c
    return -math.log(c)


def dataset_node(ds: str) -> str:
    return f"dataset:{ds}"


def column_node(ds: str, col: str) -> str:
    return f"column:{ds}::{col}"


def entity_node(entity_id: str) -> str:
    return f"entity:{entity_id}"


class IntegrationGraph:
    def __init__(self, cost_mode: str = "neglog"):
        self.g = nx.MultiGraph()
        self.cost_mode = cost_mode

    # ------------------------------------------------------------------ construction
    def add_dataset(self, artifact: DatasetArtifact, profile: DatasetProfile | None) -> None:
        self.g.add_node(
            dataset_node(artifact.dataset_id),
            kind="dataset",
            dataset_id=artifact.dataset_id,
            label=artifact.source_name,
            source_type=artifact.source_type.value,
            rows=artifact.row_count,
            columns=artifact.column_count,
        )
        if profile:
            for col in profile.columns:
                n = column_node(artifact.dataset_id, col.name)
                self.g.add_node(
                    n,
                    kind="column",
                    dataset_id=artifact.dataset_id,
                    column=col.name,
                    label=col.name,
                    semantic_type=col.semantic_type.value,
                    data_type=col.data_type.value,
                    uniqueness=col.uniqueness,
                )
                self._edge(dataset_node(artifact.dataset_id), n, "contains", 1.0, {})

    def add_column_matches(self, matches: list[ColumnMatch], include_rejected: bool = False) -> None:
        for m in matches:
            if not m.accepted and not include_rejected:
                continue
            rel = m.relationship
            if rel == "similar_column" and m.evidence.value_overlap >= 0.8 and m.evidence.name_similarity < 0.6:
                rel = "value_overlap"
            elif rel == "similar_column" and m.evidence.semantic_similarity >= 0.8 and m.evidence.value_overlap < 0.2:
                rel = "semantic_match"
            self._edge(
                column_node(m.left.dataset_id, m.left.column),
                column_node(m.right.dataset_id, m.right.column),
                rel,
                m.score,
                {**m.evidence.model_dump(), "accepted": m.accepted, "band": m.band.value, "rejection_reason": m.rejection_reason},
            )

    def add_dataset_relationships(self, rels: list[DatasetRelationship]) -> None:
        for r in rels:
            self._edge(
                dataset_node(r.left_dataset),
                dataset_node(r.right_dataset),
                "dataset_relationship",
                r.confidence,
                {
                    "join_kind": r.join_kind.value,
                    "band": r.band.value,
                    "explanation": r.explanation,
                    "key_matches": [f"{m.left.key} ↔ {m.right.key}" for m in r.key_matches],
                    "attribute_matches": [f"{m.left.key} ↔ {m.right.key}" for m in r.attribute_matches],
                    **{k: v for k, v in r.evidence.items() if k != "alternatives"},
                },
            )

    def add_entity_clusters(self, clusters: list, max_entities: int = 2000) -> int:
        """Entity nodes with same_entity edges to their member datasets (capped for large results)."""
        added = 0
        for c in clusters:
            if c.size < 2 or added >= max_entities:
                continue
            en = entity_node(c.entity_id)
            self.g.add_node(en, kind="entity", entity_id=c.entity_id, size=c.size, label=c.entity_id)
            for ds in {d for d, _ in c.members}:
                self._edge(en, dataset_node(ds), "same_entity", c.confidence, {"records": sum(1 for d, _ in c.members if d == ds)})
            added += 1
        return added

    def add_derived(self, output_id: str, label: str, sources: list[str], rows: int) -> None:
        node = f"output:{output_id}"
        self.g.add_node(node, kind="output", label=label, rows=rows)
        for ds in sources:
            self._edge(node, dataset_node(ds), "derived_from", 1.0, {})

    def _edge(self, u: str, v: str, rel: str, confidence: float, evidence: dict[str, Any]) -> None:
        # replace an existing edge of the same relationship type (idempotent rebuilds)
        if self.g.has_edge(u, v):
            for k, d in list(self.g.get_edge_data(u, v).items()):
                if d["relationship_type"] == rel:
                    self.g.remove_edge(u, v, key=k)
        self.g.add_edge(
            u, v,
            relationship_type=rel,
            confidence=round(confidence, 4),
            weight=round(confidence, 4),
            cost=edge_cost(confidence, self.cost_mode),
            evidence=evidence,
        )

    # ------------------------------------------------------------------ views
    def dataset_view(self, min_confidence: float = 0.0) -> nx.Graph:
        """Simple weighted graph over datasets (strongest relationship per pair)."""
        h = nx.Graph()
        for n, d in self.g.nodes(data=True):
            if d["kind"] == "dataset":
                h.add_node(d["dataset_id"], **d)
        for u, v, d in self.g.edges(data=True):
            if d["relationship_type"] != "dataset_relationship" or d["confidence"] < min_confidence:
                continue
            a, b = self.g.nodes[u]["dataset_id"], self.g.nodes[v]["dataset_id"]
            if not h.has_edge(a, b) or h.edges[a, b]["confidence"] < d["confidence"]:
                h.add_edge(a, b, **d)
        return h

    def column_view(self) -> nx.Graph:
        h = nx.Graph()
        for n, d in self.g.nodes(data=True):
            if d["kind"] == "column":
                h.add_node(n, **d)
        for u, v, d in self.g.edges(data=True):
            if self.g.nodes[u]["kind"] == "column" and self.g.nodes[v]["kind"] == "column":
                if not h.has_edge(u, v) or h.edges[u, v]["confidence"] < d["confidence"]:
                    h.add_edge(u, v, **d)
        return h

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        rels: dict[str, int] = {}
        for _, _, d in self.g.edges(data=True):
            rels[d["relationship_type"]] = rels.get(d["relationship_type"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), "node_kinds": kinds, "edge_types": rels, "cost_mode": self.cost_mode}
