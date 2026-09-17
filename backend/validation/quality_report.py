"""Data quality report (structured + human-readable)."""
from __future__ import annotations

from typing import Any

from backend.core.models import ColumnMatch, ConfidenceBand, DatasetArtifact, DatasetProfile, MergePlan
from backend.merge.executor import ExecutionContext


def build_quality_report(
    plan: MergePlan,
    artifacts: dict[str, DatasetArtifact],
    profiles: dict[str, DatasetProfile],
    matches: list[ColumnMatch],
    ctx: ExecutionContext,
    output_rows: int,
    validation: dict[str, Any],
    probability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [m for m in matches if m.accepted and m.left.dataset_id in plan.datasets and m.right.dataset_id in plan.datasets]
    bands = {b.value: sum(1 for m in accepted if m.band == b) for b in ConfidenceBand}
    rows_ingested = sum(artifacts[d].row_count for d in plan.datasets)

    entity_records = sum(s["records"] for s in ctx.group_stats.values())
    entities = sum(s["entities"] for s in ctx.group_stats.values())
    multi_source = sum(s["multi_source_entities"] for s in ctx.group_stats.values())
    within_dups = sum(s["within_dataset_duplicates_resolved"] for s in ctx.group_stats.values())
    linked_records = 0
    for gid, st in ctx.group_stats.items():
        er = ctx.er_results.get(gid)
        if er:
            linked_records += sum(c.size for c in er.clusters if c.size > 1)
    unmatched_entity_records = entity_records - linked_records if ctx.er_results else 0

    joins = ctx.join_stats
    fully_matched = None
    if joins and not ctx.spec.entity_grain:
        root_joins = [j for j in joins if j["parent"] == plan.root_dataset]
        if root_joins:
            fully_matched = min(j["matched_rows"] for j in root_joins)
    coverage_parts = [j["match_rate"] for j in joins]
    if ctx.er_results and entity_records:
        coverage_parts.append(linked_records / entity_records)
    coverage = sum(coverage_parts) / len(coverage_parts) if coverage_parts else 1.0

    exact_dups = {d: profiles[d].duplicate_rows for d in plan.datasets if profiles[d].duplicate_rows}
    report = {
        "datasets": len(plan.datasets),
        "unmerged_datasets": plan.unmerged_datasets,
        "rows_ingested": rows_ingested,
        "output_rows": output_rows,
        "grain": plan.grain,
        "schema_mappings": len(accepted),
        "schema_mappings_by_band": bands,
        "rejected_candidate_mappings": sum(1 for m in matches if not m.accepted),
        "entity_records": entity_records,
        "resolved_entities": entities,
        "multi_source_entities": multi_source,
        "linked_entity_records": linked_records,
        "unlinked_entity_records": unmatched_entity_records,
        "duplicate_entities_resolved": within_dups,
        "exact_duplicate_rows": exact_dups,
        "conflicts": len(ctx.conflicts),
        "conflicts_by_attribute": _count_by(ctx.conflicts, "attribute"),
        "joins": [{k: j[k] for k in ("parent", "child", "kind", "role", "matched_rows", "unmatched_rows", "match_rate")} for j in joins],
        "rows_matched_in_all_lookups": fully_matched,
        "integration_coverage": round(coverage, 4),
        "expected_correct_rows": (probability or {}).get("expected_correct_rows"),
        "rows_below_auto_threshold": (probability or {}).get("rows_below_threshold"),
        "validation_passed": validation["passed"],
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
    }
    report["text"] = render_text(report)
    return report


def _count_by(items, attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        k = getattr(it, attr)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def render_text(r: dict[str, Any]) -> str:
    lines = [
        "DATA QUALITY REPORT",
        "",
        f"Datasets: {r['datasets']}" + (f" (+{len(r['unmerged_datasets'])} left unmerged)" if r["unmerged_datasets"] else ""),
        f"Rows ingested: {r['rows_ingested']:,}",
        f"Unified rows: {r['output_rows']:,} ({r['grain']})",
        "",
        f"Schema mappings: {r['schema_mappings']}",
        f"  High confidence: {r['schema_mappings_by_band']['HIGH']}",
        f"  Medium confidence: {r['schema_mappings_by_band']['MEDIUM']}",
        f"  Low confidence: {r['schema_mappings_by_band']['LOW']}",
        "",
    ]
    if r["entity_records"]:
        lines += [
            f"Entity records: {r['entity_records']:,} → resolved entities: {r['resolved_entities']:,}",
            f"  Matched (linked) entity records: {r['linked_entity_records']:,}",
            f"  Unmatched entity records: {r['unlinked_entity_records']:,}",
            f"  Entities present in several sources: {r['multi_source_entities']:,}",
            f"Duplicate entities resolved: {r['duplicate_entities_resolved']:,}",
            "",
        ]
    for j in r["joins"]:
        role = f" as {j['role']}" if j["role"] else ""
        lines.append(f"Join {j['parent']} ← {j['child']}{role} ({j['kind']}): {j['matched_rows']:,} matched, {j['unmatched_rows']:,} unmatched ({j['match_rate']:.1%})")
    if r["joins"]:
        lines.append("")
    if r.get("expected_correct_rows") is not None:
        lines += [f"Expected correctly integrated rows (Σ _match_probability): {r['expected_correct_rows']:,.1f} of {r['output_rows']:,}",
                  f"  Rows resting on a link below the auto-merge threshold: {r['rows_below_auto_threshold']:,}", ""]
    lines += [
        f"Conflicts: {r['conflicts']:,}",
        f"Overall integration coverage: {r['integration_coverage']:.1%}",
        f"Validation: {'PASSED' if r['validation_passed'] else 'FAILED'} ({r['validation_errors']} errors, {r['validation_warnings']} warnings)",
    ]
    return "\n".join(lines)
