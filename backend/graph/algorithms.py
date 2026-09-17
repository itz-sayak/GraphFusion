"""Graph algorithms over the integration graph — each used where it fits.

| Problem                                   | Algorithm                         | Why                                                   |
|-------------------------------------------|-----------------------------------|-------------------------------------------------------|
| Which datasets can be integrated at all   | connected components              | reachability under confident relationships            |
| Consistent merge plan                     | maximum spanning tree (Kruskal)   | n−1 joins, acyclic (no double counting), max total c  |
| Best route between two datasets/columns   | Dijkstra on cost = −log c         | maximises path reliability Π c(e)                     |
| Top alternative routes                    | Yen's k-shortest simple paths     | explain competing integration routes                  |
| 1:1 column alignment                      | Hungarian / max-weight matching   | global optimum under the 1:1 constraint (aligner.py)  |
| Entity clustering                         | correlation clustering (GAEC)     | avoids transitive over-merging (clustering.py)        |

Why −log c: if edges are independent pieces of evidence, the reliability of a
path is the product of its edge confidences. Because −log is monotone
decreasing, argmin_P Σ −log c(e) = argmax_P Π c(e). Dijkstra is exact here
since all costs are non-negative (c ≤ 1). The linear cost 1 − c is also
supported but has no probabilistic reading: it prefers a direct 0.45 edge
(cost 0.55) over a 3-hop path of 0.8-confidence edges (cost 0.60) although the
path is the more reliable one (0.51 > 0.45); −log c ranks them correctly
(0.80 vs 0.67).
"""
from __future__ import annotations

from itertools import islice
from typing import Any

import networkx as nx

from backend.graph.schema_graph import IntegrationGraph, column_node, edge_cost


def integration_components(graph: IntegrationGraph, min_confidence: float) -> list[list[str]]:
    h = graph.dataset_view(min_confidence)
    comps = [sorted(c) for c in nx.connected_components(h)]
    comps.sort(key=lambda c: (-len(c), c))
    return comps


def maximum_spanning_plan(graph: IntegrationGraph, min_confidence: float) -> tuple[nx.Graph, list[dict[str, Any]]]:
    """Maximum spanning forest over confident dataset relationships.

    Returns the forest and the relationships that were *excluded* because they
    would close a cycle, each with the stronger path that made it redundant.
    """
    h = graph.dataset_view(min_confidence)
    forest = nx.maximum_spanning_tree(h, weight="confidence", algorithm="kruskal")
    excluded = []
    for u, v, d in h.edges(data=True):
        if forest.has_edge(u, v):
            continue
        path = nx.shortest_path(forest, u, v) if nx.has_path(forest, u, v) else []
        path_conf = path_reliability(forest, path)
        excluded.append(
            {
                "left": u,
                "right": v,
                "confidence": d["confidence"],
                "join_kind": d["evidence"].get("join_kind"),
                "reason": "would create a cycle; datasets already connected through a stronger route",
                "alternative_path": path,
                "alternative_min_edge": min((forest.edges[a, b]["confidence"] for a, b in zip(path, path[1:])), default=None),
                "alternative_reliability": round(path_conf, 4),
            }
        )
    return forest, excluded


def path_reliability(g: nx.Graph, path: list[str]) -> float:
    r = 1.0
    for a, b in zip(path, path[1:]):
        r *= g.edges[a, b]["confidence"]
    return r if path else 0.0


def _cost_graph(h: nx.Graph, cost_mode: str) -> nx.Graph:
    for _, _, d in h.edges(data=True):
        d["cost"] = edge_cost(d["confidence"], cost_mode)
    return h


def best_route(graph: IntegrationGraph, source: str, target: str, min_confidence: float = 0.0, cost_mode: str | None = None, k: int = 3) -> dict[str, Any]:
    """Dijkstra least-cost route between two datasets, compared with the direct edge."""
    mode = cost_mode or graph.cost_mode
    h = _cost_graph(graph.dataset_view(min_confidence), mode)
    if source not in h or target not in h:
        return {"found": False, "reason": "unknown dataset"}
    direct = h.edges[source, target]["confidence"] if h.has_edge(source, target) else None
    if not nx.has_path(h, source, target):
        return {"found": False, "source": source, "target": target, "direct_confidence": direct, "reason": "no route above the confidence threshold"}
    path = nx.dijkstra_path(h, source, target, weight="cost")
    cost = nx.dijkstra_path_length(h, source, target, weight="cost")
    hops = [
        {
            "from": a,
            "to": b,
            "confidence": h.edges[a, b]["confidence"],
            "cost": round(h.edges[a, b]["cost"], 4),
            "join_kind": h.edges[a, b]["evidence"].get("join_kind"),
            "keys": h.edges[a, b]["evidence"].get("key_matches", []),
        }
        for a, b in zip(path, path[1:])
    ]
    alternatives = []
    for p in islice(nx.shortest_simple_paths(h, source, target, weight="cost"), k):
        alternatives.append({"path": p, "cost": round(sum(h.edges[a, b]["cost"] for a, b in zip(p, p[1:])), 4), "reliability": round(path_reliability(h, p), 4)})
    return {
        "found": True,
        "source": source,
        "target": target,
        "cost_mode": mode,
        "path": path,
        "hops": hops,
        "total_cost": round(cost, 4),
        "reliability": round(path_reliability(h, path), 4),
        "direct_confidence": direct,
        "uses_intermediate": len(path) > 2,
        "alternatives": alternatives,
    }


def column_route(graph: IntegrationGraph, source: tuple[str, str], target: tuple[str, str], cost_mode: str | None = None) -> dict[str, Any]:
    """Dijkstra over the column graph: how is column A.x connected to column C.z?"""
    mode = cost_mode or graph.cost_mode
    h = _cost_graph(graph.column_view(), mode)
    s, t = column_node(*source), column_node(*target)
    if s not in h or t not in h or not nx.has_path(h, s, t):
        return {"found": False}
    path = nx.dijkstra_path(h, s, t, weight="cost")
    return {
        "found": True,
        "path": [h.nodes[n]["dataset_id"] + "." + h.nodes[n]["column"] for n in path],
        "reliability": round(path_reliability(h, path), 4),
        "hops": [{"confidence": h.edges[a, b]["confidence"], "relationship": h.edges[a, b]["relationship_type"]} for a, b in zip(path, path[1:])],
    }


def column_groups(matches: list, datasets: set[str] | None = None) -> list[set[tuple[str, str]]]:
    """Connected components over accepted column correspondences → canonical attribute groups."""
    h = nx.Graph()
    for m in matches:
        if not m.accepted:
            continue
        if datasets and (m.left.dataset_id not in datasets or m.right.dataset_id not in datasets):
            continue
        h.add_edge((m.left.dataset_id, m.left.column), (m.right.dataset_id, m.right.column), score=m.score)
    return [set(c) for c in nx.connected_components(h)]
