"""Merge planner: integration graph → explicit, reviewable merge plan.

1. Keep dataset relationships whose confidence clears the mode's ``min_edge``.
2. Connected components → which datasets can be integrated together.
3. Maximum spanning tree per component → acyclic join structure with the
   highest total confidence (redundant, weaker relationships are listed as
   excluded together with the stronger route that replaces them).
4. Root = the grain that keeps the most data un-aggregated: for every
   candidate, orient the tree from it and count the datasets reachable
   through N:1 lookups and same-entity joins alone (no aggregation on the
   way). The root maximises that count; ties go to a dataset that is never a
   lookup dimension, then to the larger one. A star schema's fact table wins
   (it references every dimension), a trip table stays at trip grain, and a
   large reference table that only joins through aggregation (e.g. zip-code
   geolocation points) is never chosen merely for its size. A user-pinned
   primary identifier overrides the choice.
5. Orient the tree from the root. A lookup whose fact side is the *child*
   would change the grain, so it becomes an aggregation (child aggregated to
   the key before joining). Entity-resolution / shared-key edges form entity
   groups fused at their member closest to the root (the anchor).
6. Plan per-dataset value transformations, then project the exact output
   schema (with lineage) from the plan.
"""
from __future__ import annotations

import uuid
from collections import deque
from typing import Any

import networkx as nx

from backend.core.errors import InvalidStateError, ToolArgumentError
from backend.core.models import (
    CanonicalColumn, ColumnMatch, ColumnRef, DataType, DatasetArtifact, DatasetProfile, DatasetRelationship,
    JoinKind, MergePlan, MergeStep, SemanticType,
)
from backend.graph import algorithms as ga
from backend.graph.schema_graph import IntegrationGraph
from backend.logging_conf import get_logger
from backend.merge.canonical import canonical_attribute_name, role_for
from backend.merge.modes import effective_thresholds
from backend.merge.projection import ProjectionBuilder
from backend.merge.transforms import plan_transformations

log = get_logger(__name__)

CONFLICT_STRATEGIES = ("majority_vote", "prefer_source", "prefer_latest", "prefer_non_null", "highest_confidence", "source_accuracy_vote", "keep_all", "manual_review")
_NON_FUSABLE = {SemanticType.CURRENCY}


def _rel_index(rels: list[DatasetRelationship]) -> dict[frozenset, DatasetRelationship]:
    return {frozenset((r.left_dataset, r.right_dataset)): r for r in rels}


def _pk_dataset(primary_key: str | None, profiles: dict[str, DatasetProfile]) -> tuple[str, str] | None:
    if not primary_key:
        return None
    if "." in primary_key or "::" in primary_key:
        ds, col = primary_key.replace("::", ".").split(".", 1)
        if ds in profiles and any(c.name == col for c in profiles[ds].columns):
            return ds, col
    hits = [(ds, c.name) for ds, p in profiles.items() for c in p.columns if c.name.lower() == primary_key.lower()]
    if not hits:
        raise ToolArgumentError(f"Primary identifier {primary_key!r} does not exist in any loaded dataset")
    return hits[0]


def _preserved_datasets(forest: nx.Graph, rel_by_pair: dict, root: str) -> int:
    """Datasets whose rows join a plan rooted at ``root`` without being aggregated.

    An edge keeps the grain when it is a same-entity join, or a lookup whose fact (many) side is the parent.
    Reverse lookups and aggregated lookups collapse the child to the parent's key, and everything below them.
    """
    seen, stack = {root}, [root]
    while stack:
        parent = stack.pop()
        for child in forest.neighbors(parent):
            if child in seen:
                continue
            r = rel_by_pair[frozenset((parent, child))]
            keeps = r.join_kind in (JoinKind.ENTITY_RESOLUTION, JoinKind.ENTITY_KEY_MERGE) or (
                r.join_kind == JoinKind.LOOKUP and r.evidence.get("fact") == parent)
            if keeps:
                seen.add(child)
                stack.append(child)
    return len(seen)


def build_plan(
    artifacts: dict[str, DatasetArtifact],
    profiles: dict[str, DatasetProfile],
    matches: list[ColumnMatch],
    rels: list[DatasetRelationship],
    graph: IntegrationGraph,
    config: dict,
    mode: str = "balanced",
    conflict_strategy: str = "majority_vote",
    overrides: dict[str, Any] | None = None,
) -> MergePlan:
    overrides = dict(overrides or {})
    if conflict_strategy not in CONFLICT_STRATEGIES:
        raise ToolArgumentError(f"Unknown conflict strategy {conflict_strategy!r}; expected one of {CONFLICT_STRATEGIES}")
    datasets = [d for d in artifacts if d in profiles]
    if overrides.get("datasets"):
        datasets = [d for d in datasets if d in set(overrides["datasets"])]
    if not datasets:
        raise InvalidStateError("No profiled datasets to plan a merge for")
    t = effective_thresholds(config, mode, overrides)
    vendor = tuple(config["merge"].get("vendor_prefixes", []))
    rel_by_pair = _rel_index(rels)
    warnings: list[str] = []

    h = graph.dataset_view(0.0).subgraph(datasets).copy()
    excluded: list[dict[str, Any]] = []
    for u, v, d in list(h.edges(data=True)):
        if d["confidence"] < t["min_edge"]:
            excluded.append({"left": u, "right": v, "confidence": d["confidence"], "join_kind": d["evidence"].get("join_kind"), "reason": f"confidence {d['confidence']:.2f} below the {mode} threshold {t['min_edge']:.2f}"})
            h.remove_edge(u, v)

    components = sorted((sorted(c) for c in nx.connected_components(h)), key=lambda c: (-len(c), -sum(artifacts[d].row_count for d in c)))
    pk = _pk_dataset(overrides.get("primary_key"), profiles)
    main = next((c for c in components if pk and pk[0] in c), components[0])
    unmerged = [d for c in components if c is not main for d in c]
    for d in unmerged:
        warnings.append(f"{d} has no relationship above the {mode} threshold with the main integration component; it is left unmerged (not discarded).")

    forest = nx.maximum_spanning_tree(h.subgraph(main), weight="confidence", algorithm="kruskal")
    for u, v, d in h.subgraph(main).edges(data=True):
        if not forest.has_edge(u, v):
            path = nx.shortest_path(forest, u, v)
            excluded.append({
                "left": u, "right": v, "confidence": d["confidence"], "join_kind": d["evidence"].get("join_kind"),
                "reason": "redundant: would close a cycle (double-count rows); datasets already connected by a stronger route",
                "alternative_path": path, "alternative_reliability": round(ga.path_reliability(forest, path), 4),
            })

    # ------------------------------------------------------------ root selection
    dims: set[str] = set()
    for u, v in forest.edges():
        r = rel_by_pair[frozenset((u, v))]
        if r.join_kind in (JoinKind.LOOKUP, JoinKind.AGGREGATE_LOOKUP):
            dims.add(r.evidence["dimension"])
    if pk:
        root = pk[0]
    else:
        root = max(main, key=lambda d: (_preserved_datasets(forest, rel_by_pair, d), d not in dims, artifacts[d].row_count, d))

    # ------------------------------------------------------------ orientation
    depth = {root: 0}
    order = deque([root])
    tree: list[dict[str, Any]] = []
    group_uf = nx.Graph()
    group_uf.add_nodes_from(main)
    while order:
        parent = order.popleft()
        for child in sorted(forest.neighbors(parent)):
            if child in depth:
                continue
            depth[child] = depth[parent] + 1
            order.append(child)
            r = rel_by_pair[frozenset((parent, child))]
            requires_review = r.confidence < t["auto_merge"]
            if r.join_kind in (JoinKind.ENTITY_RESOLUTION, JoinKind.ENTITY_KEY_MERGE):
                group_uf.add_edge(parent, child, kind=r.join_kind.value)
                tree.append({"parent": parent, "child": child, "join_kind": r.join_kind.value, "confidence": r.confidence, "key_pairs": [], "requires_review": requires_review,
                             "explanation": r.explanation})
                continue
            keys = []
            fact = r.evidence.get("fact")
            for m in r.key_matches:
                pcol = m.left.column if m.left.dataset_id == parent else m.right.column
                ccol = m.right.column if m.right.dataset_id == child else m.left.column
                keys.append({"parent_column": pcol, "child_column": ccol, "normalizer": m.evidence.normalizer, "match_confidence": m.score})
            if r.join_kind == JoinKind.LOOKUP and fact == parent:
                kind = "lookup"
                for k in keys:
                    k["role"] = role_for(k["parent_column"], k["child_column"], vendor)
            elif r.join_kind == JoinKind.LOOKUP:
                kind = "reverse_aggregate"
                warnings.append(f"{child} is finer-grained than {parent}; it is aggregated to {parent}'s key to preserve the {root} grain.")
                for k in keys:
                    k["role"] = role_for(k["child_column"], k["parent_column"], vendor) if len(keys) > 1 else None
            else:
                kind = "aggregate_lookup"
                for k in keys:
                    k["role"] = None
            tree.append({"parent": parent, "child": child, "join_kind": kind, "confidence": r.confidence, "key_pairs": keys, "requires_review": requires_review, "explanation": r.explanation})

    # ------------------------------------------------------------ entity groups
    groups: list[dict[str, Any]] = []
    for gi, comp in enumerate(c for c in nx.connected_components(group_uf) if len(c) > 1):
        members = sorted(comp, key=lambda d: (depth[d], d))
        anchor = members[0]
        kinds = {d["kind"] for _, _, d in group_uf.subgraph(comp).edges(data=True)}
        method = "key" if kinds == {JoinKind.ENTITY_KEY_MERGE.value} else "fellegi_sunter"
        fields = _group_fields(comp, members, matches, profiles, rels, vendor)
        if not fields:
            warnings.append(f"Entity group {sorted(comp)} has no fusable attributes; records are kept separate.")
        groups.append({"group_id": f"G{gi + 1}", "datasets": members, "anchor": anchor, "method": method, "fields": fields,
                       "auto_threshold": t.get("entity_merge", t["auto_merge"]), "review_threshold": t.get("entity_review", t["review"]), "dedupe": members})

    # a dataset that is a lone ER target with no fusable fields is dropped from the group structure
    transforms = {d: plan_transformations(profiles[d], config["merge"]["base_currency"]) for d in main}

    plan = MergePlan(
        plan_id=f"plan_{uuid.uuid4().hex[:10]}",
        mode=mode,
        conflict_strategy=conflict_strategy,
        thresholds=t,
        root_dataset=root,
        grain=f"one row per {root} record" if not any(g["anchor"] == root for g in groups) else f"one row per resolved entity of {', '.join(next(g['datasets'] for g in groups if g['anchor'] == root))}",
        datasets=main,
        steps=[],
        canonical_schema=[],
        excluded_relationships=excluded,
        integration_tree=tree,
        entity_groups=groups,
        unmerged_datasets=unmerged,
        routes=[],
        transformations=transforms,
        overrides=overrides,
        warnings=warnings,
    )
    spec = ProjectionBuilder(tree, groups, profiles, transforms, vendor).build(root)
    plan.canonical_schema = [
        CanonicalColumn(
            name=c.output,
            semantic_type=SemanticType(c.semantic_type),
            data_type=DataType(c.data_type),
            sources=[ColumnRef(dataset_id=d, column=col) for d, col in (c.members or [(c.source_dataset, c.source_column)]) if not col.startswith("__")],
            role=c.operation,
            transformation=c.transformation or c.aggregate,
        )
        for c in spec.columns
    ]
    plan.steps = _describe_steps(plan, profiles)
    for d in main:
        if d == root:
            continue
        route = ga.best_route(graph, root, d, min_confidence=t["min_edge"])
        if route.get("found"):
            plan.routes.append({"target": d, "path": route["path"], "reliability": route["reliability"], "direct_confidence": route["direct_confidence"]})
    log.info("Merge plan generated", plan_id=plan.plan_id, root=root, datasets=len(main), steps=len(plan.steps), entity_groups=len(groups), excluded=len(excluded), mode=mode)
    return plan


def _group_fields(comp: set[str], members: list[str], matches: list[ColumnMatch], profiles: dict[str, DatasetProfile], rels: list[DatasetRelationship], vendor: tuple[str, ...]) -> list[dict[str, Any]]:
    groups = ga.column_groups(matches, set(comp))
    fields = []
    used: set[tuple[str, str]] = set()
    for g in sorted(groups, key=lambda g: sorted(g)):
        by_ds: dict[str, str] = {}
        for ds, col in sorted(g, key=lambda x: (members.index(x[0]), x[1])):
            if ds not in by_ds and (ds, col) not in used:
                by_ds[ds] = col
        if len(by_ds) < 2:
            continue
        sem_types = [profiles[d].column(c).semantic_type for d, c in by_ds.items()]
        sem = max(set(sem_types), key=sem_types.count)
        if sem in _NON_FUSABLE:
            continue
        best = max((m for m in matches if m.accepted and (m.left.dataset_id, m.left.column) in g and (m.right.dataset_id, m.right.column) in g), key=lambda m: m.score)
        dtypes = [profiles[d].column(c).data_type for d, c in by_ds.items()]
        used |= set(by_ds.items())
        fields.append({
            "name": canonical_attribute_name(list(by_ds.items()), members, vendor),
            "output_name": canonical_attribute_name(list(by_ds.items()), members, vendor),
            "semantic_type": sem.value,
            "data_type": dtypes[0].value if len(set(dtypes)) == 1 else DataType.STRING.value,
            "columns": by_ds,
            "normalizer": best.evidence.normalizer,
            "match_confidence": best.score,
        })
    return fields


def _describe_steps(plan: MergePlan, profiles: dict[str, DatasetProfile]) -> list[MergeStep]:
    steps: list[MergeStep] = []
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        steps.append(MergeStep(step=n, **kw))

    for ds, ts in plan.transformations.items():
        if ts:
            add(operation="transform", description=f"Normalise {ds}: " + "; ".join(f"{x['column']} → {x['description']}" for x in ts), left=ds, transformations=[f"{x['column']}:{x['transform']}" for x in ts])
    for ds in plan.datasets:
        dups = profiles[ds].duplicate_rows
        if dups:
            add(operation="flag_duplicates", description=f"{ds} contains {dups} exact duplicate rows; they stay in the source view, are resolved to one entity and reported (nothing is deleted).", left=ds)
    for g in plan.entity_groups:
        fields = ", ".join(f"{f['output_name']} ⇐ {' / '.join(f'{d}.{c}' for d, c in f['columns'].items())} ('{f['normalizer']}')" for f in g["fields"])
        method = "shared-key grouping" if g["method"] == "key" else "Fellegi–Sunter probabilistic linkage with blocking and correlation clustering (deduplicating within each dataset as well)"
        add(
            operation="resolve_entities",
            description=f"Resolve entities across {', '.join(g['datasets'])} using {method}. Compared attributes: {fields}. Links ≥ {g['auto_threshold']:.2f} merge automatically; {g['review_threshold']:.2f}–{g['auto_threshold']:.2f} are flagged for review and not merged.",
            left=g["anchor"],
            right=", ".join(d for d in g["datasets"] if d != g["anchor"]),
            join_kind=JoinKind.ENTITY_RESOLUTION if g["method"] != "key" else JoinKind.ENTITY_KEY_MERGE,
            threshold=g["auto_threshold"],
        )
        add(
            operation="fuse_entities",
            description=f"Fuse each entity into one record at anchor {g['anchor']} with conflict strategy '{plan.conflict_strategy}'; every conflicting value is kept in the conflict report with its source.",
            left=g["anchor"],
            join_type="full_outer",
        )
    # attachments bottom-up (deepest first) = execution order
    depth = {plan.root_dataset: 0}
    for e in plan.integration_tree:
        depth[e["child"]] = depth.get(e["parent"], 0) + 1
    for e in sorted(plan.integration_tree, key=lambda e: -depth[e["child"]]):
        if e["join_kind"] in ("entity_resolution", "entity_key_merge"):
            continue
        keys = [{"parent": f"{e['parent']}.{k['parent_column']}", "child": f"{e['child']}.{k['child_column']}", "normalizer": k["normalizer"], "role": k.get("role") or ""} for k in e["key_pairs"]]
        if e["join_kind"] == "lookup":
            roles = [k["role"] for k in keys if k["role"]]
            desc = f"LEFT JOIN {e['child']} onto {e['parent']} via " + ", ".join(f"{k['parent']} = {k['child']} ('{k['normalizer']}')" for k in keys)
            if roles:
                desc += f" — once per role ({', '.join(roles)}), prefixing attributes"
            desc += f". {e['child']} is deduplicated on the key first so {e['parent']} rows are never multiplied."
            jt = "left"
        else:
            desc = f"Aggregate {e['child']} to the key " + ", ".join(k["child"] for k in keys) + f" (sum/avg/mode per column), then LEFT JOIN onto {e['parent']}."
            jt = "left"
        add(operation=e["join_kind"], description=desc, left=e["parent"], right=e["child"], join_kind=JoinKind.LOOKUP if e["join_kind"] == "lookup" else JoinKind.AGGREGATE_LOOKUP,
            join_type=jt, join_keys=keys, confidence=e["confidence"], threshold=plan.thresholds["min_edge"], requires_review=e["requires_review"])
    add(operation="preserve_provenance", description="Record column lineage, per-row source row ids (_src_*), per-join match flags (_match_*) and cell-level provenance of fused values.")
    add(operation="validate", description=f"Validate: output rows = {plan.root_dataset} rows (no fan-out, no silent drops), source checksums unchanged, every record assigned to exactly one entity.")
    add(operation="materialize", description="Write unified_dataset.parquet/csv and reports.")
    return steps
