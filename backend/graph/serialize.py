"""Graph serialisation: ``integration_graph.json`` and React-Flow-ready views."""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

from backend.graph.schema_graph import IntegrationGraph


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return v


def to_json(graph: IntegrationGraph) -> dict[str, Any]:
    nodes = [{"id": n, **_jsonable(d)} for n, d in graph.g.nodes(data=True)]
    edges = []
    for i, (u, v, d) in enumerate(graph.g.edges(data=True)):
        edges.append({"id": f"e{i}", "source": u, "target": v, **_jsonable(d)})
    return {"stats": graph.stats(), "nodes": nodes, "edges": edges}


def react_flow_view(graph: IntegrationGraph, level: str = "dataset", tree_edges: set[frozenset] | None = None) -> dict[str, Any]:
    """Nodes/edges with positions.

    ``level='dataset'``: datasets and their relationships (plus the chosen merge tree).
    ``level='column'``: datasets as groups with their matched columns and correspondences.
    """
    tree_edges = tree_edges or set()
    if level == "dataset":
        h = graph.dataset_view()
        pos = nx.circular_layout(h, scale=320) if h.number_of_nodes() > 1 else {n: (0, 0) for n in h}
        nodes = [
            {
                "id": n,
                "type": "dataset",
                "position": {"x": float(pos[n][0]) + 400, "y": float(pos[n][1]) + 340},
                "data": {"label": d.get("label", n), "rows": d.get("rows"), "columns": d.get("columns"), "source_type": d.get("source_type")},
            }
            for n, d in h.nodes(data=True)
        ]
        edges = []
        for u, v, d in h.edges(data=True):
            ev = d["evidence"]
            edges.append(
                {
                    "id": f"{u}--{v}",
                    "source": u,
                    "target": v,
                    "label": f"{ev.get('join_kind', '')} · {d['confidence']:.2f}",
                    "data": _jsonable({"confidence": d["confidence"], "relationship_type": d["relationship_type"], "in_merge_tree": frozenset((u, v)) in tree_edges, **ev}),
                }
            )
        return {"level": level, "nodes": nodes, "edges": edges}

    col_graph = graph.column_view()
    matched = {n for n in col_graph if col_graph.degree(n) > 0}
    ds_of = {n: graph.g.nodes[n]["dataset_id"] for n in matched}
    # hub (most correspondences) in the middle, then alternate sides — keeps edges short
    degree: dict[str, int] = {}
    for n in matched:
        degree[ds_of[n]] = degree.get(ds_of[n], 0) + col_graph.degree(n)
    ranked = sorted(degree, key=lambda d: (-degree[d], d))
    datasets: list[str] = []
    for i, d in enumerate(ranked):
        datasets.insert(0, d) if i % 2 else datasets.append(d)
    order: dict[str, list[str]] = {d: sorted(n for n in matched if ds_of[n] == d) for d in datasets}
    # barycenter pass: order each group's columns by the mean position of their partners in the hub group
    hub = ranked[0] if ranked else None
    if hub:
        pos = {n: i for i, n in enumerate(order[hub])}
        for d in datasets:
            if d == hub:
                continue
            def bary(n):
                ps = [pos[m] for m in col_graph.neighbors(n) if m in pos]
                return (sum(ps) / len(ps)) if ps else 1e9
            order[d].sort(key=lambda n: (bary(n), n))
    nodes, edges = [], []
    for i, ds in enumerate(datasets):
        cols = order[ds]
        x = i * 380
        nodes.append({"id": f"group:{ds}", "type": "group", "position": {"x": x, "y": 0}, "data": {"label": ds}, "style": {"width": 300, "height": 70 + 56 * len(cols)}})
        for j, n in enumerate(cols):
            d = graph.g.nodes[n]
            nodes.append(
                {
                    "id": n,
                    "type": "column",
                    "parentId": f"group:{ds}",
                    "extent": "parent",
                    "position": {"x": 20, "y": 50 + j * 56},
                    "data": {"label": d["column"], "semantic_type": d["semantic_type"], "data_type": d["data_type"], "dataset_id": ds},
                }
            )
    col_index = {ds: i for i, ds in enumerate(datasets)}
    for u, v, d in col_graph.edges(data=True):
        if col_index[ds_of[u]] > col_index[ds_of[v]]:
            u, v = v, u  # draw left → right: source handle on the right edge of the left group
        edges.append(
            {
                "id": f"{u}--{v}",
                "source": u,
                "target": v,
                "label": f"{d['confidence']:.2f}",
                "data": _jsonable({"confidence": d["confidence"], "relationship_type": d["relationship_type"], "evidence": d["evidence"],
                                   "left": graph.g.nodes[u]["dataset_id"] + "." + graph.g.nodes[u]["column"],
                                   "right": graph.g.nodes[v]["dataset_id"] + "." + graph.g.nodes[v]["column"]}),
            }
        )
    return {"level": level, "nodes": nodes, "edges": edges}
