"""Provenance: column lineage, row lineage, cell lineage and a W3C PROV document.

Three granularities, chosen for what is practical at scale:

* **column lineage** — every output column traced through lookups, aggregations
  and fusion to its ultimate source column(s), with the transformation and the
  confidence of each hop (``provenance.json``);
* **row lineage** — every output row carries ``_record_id`` plus ``_src_<alias>_row``
  columns holding the row index of each source record used (stored *in* the
  unified dataset, so it scales with the data rather than as a side file);
* **cell lineage** — for fused entity attributes backed by more than one record,
  which source record supplied the chosen value and why (``cell_provenance.parquet``).

The PROV document models datasets and outputs as ``prov:Entity``, pipeline
stages as ``prov:Activity`` and the system/session as ``prov:Agent``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend import __version__
from backend.core.models import DatasetArtifact, MergePlan
from backend.merge.projection import ColumnSpec, NodeSpec


def _hop(c: ColumnSpec, operation: str, **extra: Any) -> dict[str, Any]:
    return {"operation": operation, "dataset": c.source_dataset, "column": c.source_column, "transformation": c.transformation, **extra}


def column_lineage(spec: NodeSpec, plan: MergePlan) -> dict[str, dict[str, Any]]:
    """Map output column → {sources: [...], path: [...]} by walking the projection."""
    strategy = plan.conflict_strategy
    group_by_id = {g["group_id"]: g for g in plan.entity_groups}

    def trace(node: NodeSpec, col: ColumnSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (ultimate sources, hop path) for a column of ``node``."""
        if col.operation == "select":
            src = {"dataset": node.dataset, "column": col.source_column if not col.source_column.endswith("_raw") else col.source_column[:-4], "transformation": col.transformation}
            return [src], [_hop(col, "select")]
        for att in node.attachments:
            if col in att.columns:
                keys = ", ".join(f"{node.dataset}.{k['parent_column']} = {att.child.dataset}.{k['child_column']} ({k['normalizer']})" for k in att.keys)
                if col.child_view_name is None and col.source_column in ("__matched__", "__count__"):
                    return [{"dataset": att.child.dataset, "column": "*", "transformation": col.aggregate or "match_flag"}], [
                        {"operation": att.kind, "dataset": att.child.dataset, "column": "*", "join_keys": keys, "role": att.role, "confidence": att.confidence}
                    ]
                child_col = next((cc for cc in att.child.columns if cc.view_name == col.child_view_name), None)
                if child_col is None:
                    return [], []
                if child_col.source_column in ("__matched__", "__count__"):
                    inner_att = next((a for a in att.child.attachments if child_col in a.columns), None)
                    ds = inner_att.child.dataset if inner_att else att.child.dataset
                    what = "record count" if child_col.source_column == "__count__" else "match flag"
                    return [{"dataset": ds, "column": "*", "transformation": what}], [
                        {"operation": att.kind, "dataset": att.child.dataset, "column": child_col.output, "join_keys": keys, "role": att.role, "confidence": att.confidence},
                        {"operation": inner_att.kind if inner_att else "derived", "dataset": ds, "column": "*", "transformation": what},
                    ]
                sources, path = trace(att.child, child_col)
                hop = {"operation": att.kind, "dataset": att.child.dataset, "column": child_col.output, "join_keys": keys, "role": att.role, "confidence": att.confidence}
                if col.aggregate:
                    hop["aggregate"] = col.aggregate
                return sources, [hop] + path
        if node.group is not None and col in node.group.columns:
            g = group_by_id[node.group.group_id]
            if not col.members:
                return [{"dataset": d, "column": "*", "transformation": "entity_resolution"} for d in node.group.datasets], [
                    {"operation": "entity_resolution", "method": g["method"], "datasets": node.group.datasets, "auto_threshold": g["auto_threshold"]}
                ]
            sources: list[dict[str, Any]] = []
            path: list[dict[str, Any]] = [{"operation": "fuse", "strategy": col.aggregate or strategy, "members": [f"{d}.{c}" for d, c in col.members]}]
            for ds, view_col in col.members:
                member_node = node if ds == node.dataset else node.group.members.get(ds)
                if member_node is None:
                    continue
                inner = next((cc for cc in member_node.columns if cc.view_name == view_col and cc.operation != "fuse"), None)
                if inner is None or inner is col:
                    sources.append({"dataset": ds, "column": view_col, "transformation": None})
                    continue
                s, p = trace(member_node, inner)
                sources += s
                path += p
            return sources, path
        return [{"dataset": col.source_dataset, "column": col.source_column, "transformation": col.transformation}], []

    lineage: dict[str, dict[str, Any]] = {}
    for col in spec.columns:
        sources, path = trace(spec, col)
        lineage[col.output] = {
            "semantic_type": col.semantic_type,
            "data_type": col.data_type,
            "sources": _dedupe(sources),
            "path": path,
            "merge_operation": path[0]["operation"] if path else "select",
        }
    return lineage


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for it in items:
        k = (it.get("dataset"), it.get("column"), it.get("transformation"))
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def prov_document(
    plan: MergePlan,
    artifacts: dict[str, DatasetArtifact],
    merge_id: str,
    session_id: str,
    output_rows: int,
    group_stats: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    entity: dict[str, Any] = {}
    activity: dict[str, Any] = {}
    used, generated, derived, associated = {}, {}, {}, {}
    agent = {"dfg:system": {"prov:type": "prov:SoftwareAgent", "dfg:version": __version__}, f"dfg:session/{session_id}": {"prov:type": "prov:Person"}}

    for ds in plan.datasets:
        a = artifacts[ds]
        entity[f"dfg:source/{ds}"] = {"prov:label": a.source_name, "dfg:source_uri": a.source_uri, "dfg:format": a.source_type.value, "dfg:sha256": a.checksum, "dfg:rows": a.row_count}
        entity[f"dfg:view/{ds}"] = {"prov:label": f"normalised view of {ds}", "dfg:transformations": [t["transform"] + ":" + t["column"] for t in plan.transformations.get(ds, [])]}
        activity[f"dfg:ingest/{ds}"] = {"prov:startTime": a.ingested_at.isoformat(), "dfg:reader": a.metadata.get("reader")}
        used[f"_:u_ingest_{ds}"] = {"prov:activity": f"dfg:ingest/{ds}", "prov:entity": f"dfg:source/{ds}"}
        generated[f"_:g_view_{ds}"] = {"prov:entity": f"dfg:view/{ds}", "prov:activity": f"dfg:ingest/{ds}"}
        derived[f"_:d_view_{ds}"] = {"prov:generatedEntity": f"dfg:view/{ds}", "prov:usedEntity": f"dfg:source/{ds}"}

    activity["dfg:schema_matching"] = {"dfg:method": "weighted multi-signal scoring + similarity flooding + bipartite alignment"}
    activity[f"dfg:plan/{plan.plan_id}"] = {"prov:startTime": plan.created_at.isoformat(), "dfg:mode": plan.mode, "dfg:thresholds": plan.thresholds, "dfg:root": plan.root_dataset}
    for gid, st in group_stats.items():
        activity[f"dfg:entity_resolution/{gid}"] = {"dfg:method": st["method"], "dfg:records": st["records"], "dfg:entities": st["entities"]}
        for ds in st["datasets"]:
            used[f"_:u_er_{gid}_{ds}"] = {"prov:activity": f"dfg:entity_resolution/{gid}", "prov:entity": f"dfg:view/{ds}"}
    out = f"dfg:output/{merge_id}/unified_dataset"
    entity[out] = {"prov:label": "unified dataset", "dfg:rows": output_rows, "dfg:grain": plan.grain}
    activity[f"dfg:merge/{merge_id}"] = {"prov:endTime": now, "dfg:timings_ms": timings, "dfg:conflict_strategy": plan.conflict_strategy}
    generated["_:g_out"] = {"prov:entity": out, "prov:activity": f"dfg:merge/{merge_id}", "prov:time": now}
    for ds in plan.datasets:
        used[f"_:u_merge_{ds}"] = {"prov:activity": f"dfg:merge/{merge_id}", "prov:entity": f"dfg:view/{ds}"}
        derived[f"_:d_out_{ds}"] = {"prov:generatedEntity": out, "prov:usedEntity": f"dfg:source/{ds}"}
    associated["_:assoc"] = {"prov:activity": f"dfg:merge/{merge_id}", "prov:agent": "dfg:system", "prov:plan": f"dfg:plan/{plan.plan_id}"}
    associated["_:assoc_user"] = {"prov:activity": f"dfg:merge/{merge_id}", "prov:agent": f"dfg:session/{session_id}"}
    return {
        "prefix": {"prov": "http://www.w3.org/ns/prov#", "dfg": "urn:datafusiongraph:"},
        "entity": entity,
        "activity": activity,
        "agent": agent,
        "used": used,
        "wasGeneratedBy": generated,
        "wasDerivedFrom": derived,
        "wasAssociatedWith": associated,
    }
