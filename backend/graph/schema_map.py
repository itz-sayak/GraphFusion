"""Schema map: an ER-diagram view of *which column of which dataset connects to what*.

Everything is derived from the session's profiles, scored column matches,
inferred dataset relationships and the merge plan — nothing is specific to a
scenario, so it works for any uploaded datasets.

Output
------
tables         one per dataset: format, rows, and every column with its role
               (``key``: unique identifier · ``reference``: points into another table's key ·
               ``linked``: matched to another table's column · ``attribute``: not connected)
links          one per column correspondence, typed as
               ``join_key``   the columns a relationship joins on
               ``attribute``  same information in both tables (no join)
               ``candidate``  scored but not accepted (hidden by default in the UI)
               with the dataset relationship it belongs to, direction (referencing → referenced),
               confidence, and whether the merge plan uses it
relationships  dataset-level summary per pair (kind, cardinality, key pairs, merge-tree flag)
layout         layer and order per table (Sugiyama-style, see ``layered_layout``)

Layout (``layered_layout``)
---------------------------
1. Direct every relationship from the referencing side to the referenced side:
   lookup / aggregate lookup: fact → dimension (as inferred); entity merges and
   entity resolution have no natural direction, so the larger table points to the smaller.
2. Break cycles by dropping the weakest edge of each remaining cycle.
3. Layer = longest path from a source, so fact tables sit left and dimensions right.
4. Order tables inside each layer by the barycenter of their neighbours' positions
   (alternating left-to-right and right-to-left sweeps) to reduce crossings.
5. Each weakly connected component gets its own block; datasets with no relationship are
   listed separately as ``unconnected``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from backend.core.models import ColumnMatch, DatasetProfile, DatasetRelationship, SemanticType
from backend.graph.relationships import effective_uniqueness

CARDINALITY = {
    "lookup": "N : 1",
    "aggregate_lookup": "N : M (aggregated)",
    "entity_key_merge": "1 : 1",
    "entity_resolution": "≈ same entity",
}
# unique by accident, not identifiers: measures, free text, flags, coordinates and timestamps
_NON_KEY_TYPES = {SemanticType.CURRENCY, SemanticType.NUMERIC, SemanticType.FREE_TEXT, SemanticType.BOOLEAN, SemanticType.LATITUDE, SemanticType.LONGITUDE,
                  SemanticType.DATE, SemanticType.DATETIME}


def _rel_direction(rel: DatasetRelationship, rows: dict[str, int]) -> tuple[str, str]:
    ev = rel.evidence or {}
    fact, dim = ev.get("fact"), ev.get("dimension")
    if fact and dim and {fact, dim} == {rel.left_dataset, rel.right_dataset}:
        return fact, dim
    a, b = rel.left_dataset, rel.right_dataset
    return (a, b) if (rows.get(a, 0), b) >= (rows.get(b, 0), a) else (b, a)


def layered_layout(datasets: list[str], edges: list[tuple[str, str, float]], sweeps: int = 4) -> dict[str, dict[str, int | bool]]:
    """Layer / order / component for each dataset. ``edges`` are (from, to, confidence)."""
    g = nx.DiGraph()
    g.add_nodes_from(datasets)
    for u, v, c in edges:
        if u == v:
            continue
        if g.has_edge(u, v):
            g.edges[u, v]["c"] = max(g.edges[u, v]["c"], c)
        elif g.has_edge(v, u):
            continue
        else:
            g.add_edge(u, v, c=c)
    # 2. break cycles by dropping the weakest edge of each cycle
    while True:
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            break
        weakest = min(cycle, key=lambda e: g.edges[e[0], e[1]]["c"])
        g.remove_edge(weakest[0], weakest[1])
    # 3. longest-path layering
    layer: dict[str, int] = {}
    for n in nx.topological_sort(g):
        preds = list(g.predecessors(n))
        layer[n] = max((layer[p] + 1 for p in preds), default=0)
    out: dict[str, dict[str, int | bool]] = {}
    und = g.to_undirected()
    components = sorted((sorted(c) for c in nx.connected_components(und)), key=lambda c: (-len(c), c[0]))
    comp_index = 0
    for comp in components:
        if len(comp) == 1 and und.degree(comp[0]) == 0:
            continue
        layers: dict[int, list[str]] = defaultdict(list)
        for n in comp:
            layers[layer[n]].append(n)
        for lv in layers:
            layers[lv].sort()
        # 4. barycenter sweeps
        pos = {n: i for lv in layers for i, n in enumerate(layers[lv])}
        max_layer = max(layers)
        for s in range(sweeps):
            rng = range(1, max_layer + 1) if s % 2 == 0 else range(max_layer - 1, -1, -1)
            for lv in rng:
                ref = lv - 1 if s % 2 == 0 else lv + 1

                def bary(n: str) -> float:
                    ps = [pos[m] for m in und.neighbors(n) if layer[m] == ref]
                    return sum(ps) / len(ps) if ps else pos[n]

                layers[lv].sort(key=lambda n: (bary(n), n))
                for i, n in enumerate(layers[lv]):
                    pos[n] = i
        for lv, names in layers.items():
            for i, n in enumerate(names):
                out[n] = {"layer": lv, "order": i, "component": comp_index, "unconnected": False}
        comp_index += 1
    iso = sorted(n for n in datasets if n not in out)
    for i, n in enumerate(iso):
        out[n] = {"layer": 0, "order": i, "component": comp_index, "unconnected": True}
    return out


def schema_map_view(
    artifacts: dict[str, Any],
    profiles: dict[str, DatasetProfile],
    matches: list[ColumnMatch],
    relationships: list[DatasetRelationship],
    config: dict,
    plan: Any = None,
    candidate_min_score: float | None = None,
) -> dict[str, Any]:
    # a key shown on the map must be strictly unique; near-unique columns with repeats are references, not keys
    key_unique = float(config["merge"].get("entity_merge_uniqueness", config["merge"]["key_uniqueness"]))
    medium = float(config["thresholds"]["medium"])
    candidate_min_score = medium if candidate_min_score is None else candidate_min_score
    rows = {d: a.row_count for d, a in artifacts.items()}
    tree = set()
    if plan is not None:
        tree = {frozenset((e["parent"], e["child"])) for e in plan.integration_tree}

    # dataset relationships and the role of each matched column pair inside them
    pair_role: dict[frozenset, tuple[str, DatasetRelationship]] = {}
    rel_out = []
    dir_edges: list[tuple[str, str, float]] = []
    for r in relationships:
        if r.left_dataset not in profiles or r.right_dataset not in profiles:
            continue
        src, dst = _rel_direction(r, rows)
        dir_edges.append((src, dst, r.confidence))
        for m in r.key_matches:
            pair_role[frozenset((m.left.key, m.right.key))] = ("join_key", r)
        for m in r.attribute_matches:
            pair_role.setdefault(frozenset((m.left.key, m.right.key)), ("attribute", r))
        rel_out.append({
            "id": f"{r.left_dataset}--{r.right_dataset}",
            "from": src, "to": dst,
            "join_kind": r.join_kind.value, "cardinality": CARDINALITY.get(r.join_kind.value, ""),
            "confidence": round(r.confidence, 4), "band": r.band.value,
            "in_merge_tree": frozenset((r.left_dataset, r.right_dataset)) in tree,
            "key_pairs": [{"from": _side(m, src), "to": _side(m, dst)} for m in r.key_matches],
            "attribute_pairs": len(r.attribute_matches),
            "explanation": r.explanation,
        })

    links = []
    connected_cols: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for m in matches:
        if m.left.dataset_id not in profiles or m.right.dataset_id not in profiles:
            continue
        role_rel = pair_role.get(frozenset((m.left.key, m.right.key)))
        if role_rel:
            kind, rel = role_rel
        elif m.accepted:
            kind, rel = "attribute", None
        elif m.score >= candidate_min_score and m.status != "rejected":
            kind, rel = "candidate", None
        elif m.status == "rejected":
            kind, rel = "rejected", None
        else:
            continue
        a, b = m.left, m.right
        if rel is not None:
            src, _ = _rel_direction(rel, rows)
            if a.dataset_id != src:
                a, b = b, a
        elif (rows.get(a.dataset_id, 0), b.dataset_id) < (rows.get(b.dataset_id, 0), a.dataset_id):
            a, b = b, a
        ev = m.evidence
        links.append({
            "id": f"{a.key}--{b.key}",
            "from": {"dataset": a.dataset_id, "column": a.column},
            "to": {"dataset": b.dataset_id, "column": b.column},
            "kind": kind,
            "join_kind": rel.join_kind.value if rel else None,
            "relationship": f"{rel.left_dataset}--{rel.right_dataset}" if rel else None,
            "confidence": round(m.score, 4), "band": m.band.value, "status": m.status,
            "in_merge_tree": bool(rel) and frozenset((rel.left_dataset, rel.right_dataset)) in tree,
            "normalizer": ev.normalizer,
            "evidence": {"name": round(ev.name_similarity, 3), "values": round(ev.value_overlap, 3), "semantic": round(ev.semantic_similarity, 3),
                         "type": round(ev.datatype_similarity, 3), "pattern": round(ev.pattern_similarity, 3)},
            "row_agreement": ev.row_agreement, "rows_compared": ev.rows_compared, "rejection_reason": m.rejection_reason,
        })
        if kind in ("join_key", "attribute"):
            connected_cols[a.dataset_id][a.column].add(kind)
            connected_cols[b.dataset_id][b.column].add(kind)

    # columns that reference another table's key (the referencing side of a join key)
    references: dict[str, set[str]] = defaultdict(set)
    for ln in links:
        if ln["kind"] == "join_key" and ln["join_kind"] in ("lookup", "aggregate_lookup"):
            references[ln["from"]["dataset"]].add(ln["from"]["column"])

    from backend.config import glossary

    gloss = glossary()  # optional business glossary (config/glossary.yaml): readable names for opaque codes
    tables = []
    for ds, prof in profiles.items():
        art = artifacts.get(ds)
        cols = []
        for c in prof.columns:
            uniq = effective_uniqueness(c, prof)
            is_key = uniq >= key_unique and c.unique_count > 1 and c.null_pct < 0.01 and c.semantic_type not in _NON_KEY_TYPES
            kinds = connected_cols.get(ds, {}).get(c.name, set())
            role = "key" if is_key else ("reference" if c.name in references[ds] else ("linked" if kinds else "attribute"))
            cols.append({"name": c.name, "description": gloss.get(c.name), "semantic_type": c.semantic_type.value, "data_type": c.data_type.value, "role": role,
                         "is_key": is_key, "uniqueness": round(uniq, 4), "null_pct": round(c.null_pct, 4),
                         "connected": bool(kinds), "join_key": "join_key" in kinds})
        # connected columns first (join keys, then other links), then keys, then the rest in source order
        order = {c["name"]: i for i, c in enumerate(cols)}
        cols.sort(key=lambda c: (not c["join_key"], not c["connected"], not c["is_key"], order[c["name"]]))
        tables.append({"id": ds, "label": art.source_name if art else ds, "source_type": art.source_type.value if art else "",
                       "rows": rows.get(ds), "column_count": len(cols), "columns": cols,
                       "connected_columns": sum(1 for c in cols if c["connected"])})

    layout = layered_layout(sorted(profiles), dir_edges)
    for t in tables:
        t.update(layout[t["id"]])
    kinds = defaultdict(int)
    for ln in links:
        kinds[ln["kind"]] += 1
    return {
        "level": "schema",
        "tables": sorted(tables, key=lambda t: (t["component"], t["layer"], t["order"])),
        "links": links,
        "relationships": rel_out,
        "stats": {"tables": len(tables), "relationships": len(rel_out), "links_by_kind": dict(kinds),
                  "components": 1 + max((t["component"] for t in tables if not t["unconnected"]), default=-1),
                  "unconnected": [t["id"] for t in tables if t["unconnected"]]},
    }


def _side(m: ColumnMatch, dataset: str) -> str:
    return m.left.column if m.left.dataset_id == dataset else m.right.column
