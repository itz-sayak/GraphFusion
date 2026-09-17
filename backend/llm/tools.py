"""Tool registry: the only interface between the conversational layer and the data.

Each tool has a JSON-schema signature (sent to the LLM), a validator, an
executor bound to a :class:`Session` method, a *compact* result for the LLM
context, and a deterministic markdown renderer for the user. Numbers shown to
the user always come from the renderer, never from model text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from backend.core.errors import DFGError, ToolArgumentError
from backend.merge.modes import MODES
from backend.merge.planner import CONFLICT_STRATEGIES
from backend.logging_conf import get_logger
from backend.session.state import Session

log = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[Session, dict[str, Any]], Any]
    render: Callable[[Any], str]
    card: str | None = None  # UI card type for structured display
    mutating: bool = False

    def spec(self) -> dict[str, Any]:
        """Compact JSON schema for the model: same semantics, fewer tokens (rate-limited providers count every token)."""
        required = set(self.parameters.get("required") or [])
        props: dict[str, Any] = {}
        for name, prop in self.parameters["properties"].items():
            prop = dict(prop)
            # Models often send omitted optional arguments as explicit nulls, and providers that validate
            # tool calls server-side (e.g. Groq) reject those unless the schema allows null. The backend
            # validator drops nulls, so declaring optional parameters nullable is semantics-preserving.
            if name not in required and isinstance(prop.get("type"), str):
                prop["type"] = [prop["type"], "null"]
                if "enum" in prop:
                    prop["enum"] = [*prop["enum"], None]
            props[name] = prop
        params: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            params["required"] = sorted(required)
        return {"name": self.name, "description": self.description, "parameters": params}


def _obj(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


S = {"type": "string"}
ds_param = {"type": "string", "description": "dataset id"}


# --------------------------------------------------------------------------- renderers
def _pct(x: float) -> str:
    return f"{x:.1%}"


def r_datasets(res: list[dict[str, Any]]) -> str:
    if not res:
        return "No datasets are loaded yet. Upload files (CSV, JSON, JSONL, Parquet, XLSX) or load the sample/NYC demo."
    lines = [f"I have {len(res)} dataset{'s' if len(res) != 1 else ''} loaded:", ""]
    for d in res:
        unit = "records" if d["source_type"] in ("json", "jsonl", "rest") else "rows"
        lines.append(f"**{d['source_name']}** (`{d['dataset_id']}`, {d['source_type']}) — {d['row_count']:,} {unit}, {d['column_count']} {'fields' if unit == 'records' else 'columns'}")
    return "\n".join(lines)


def r_profile(res: dict[str, Any]) -> str:
    p = res["profile"]
    lines = [f"**{res['source_name']}** — {p['row_count']:,} rows × {p['column_count']} columns, {p['duplicate_rows']} exact duplicate rows", "",
             "| column | type | semantic type | nulls | unique | sample |", "|---|---|---|---|---|---|"]
    for c in p["columns"]:
        sample = ", ".join(str(v) for v in c["sample_values"][:3])
        lines.append(f"| {c['name']} | {c['data_type']} | {c['semantic_type']} ({c['semantic_confidence']:.2f}) | {c['null_pct']:.1%} | {c['unique_count']:,} | {sample[:60]} |")
    return "\n".join(lines)


def r_discovery(res: dict[str, Any]) -> str:
    lines = [f"I detected {len(res['datasets'])} datasets.", ""]
    for d in res["datasets"]:
        unit = "records" if d["format"] in ("json", "jsonl", "rest") else "rows"
        lines += [f"**{d['source_name']}**", f"- {d['rows']:,} {unit}", f"- {d['columns']} {'fields' if unit == 'records' else 'columns'}", ""]
    if res["relationships"]:
        lines.append("**Potential relationships discovered:**")
        for r in res["relationships"]:
            lines.append(f"- {r['left']} ↔ {r['right']} — confidence **{r['confidence']:.2f}** ({r['band']}, {r['join_kind'].replace('_', ' ')}; keys: {', '.join(r['keys']) or 'attribute correspondences'})")
    else:
        lines.append("No relationships cleared the confidence threshold.")
    lines += ["", "**Likely mappings:**"]
    for m in res["mappings"]:
        l, r = m["left"].split("::")[1], m["right"].split("::")[1]
        lines.append(f"- {l} ↔ {r} ({m['confidence']:.2f}{', values via ' + m['normalizer'] if m['normalizer'] not in ('basic', 'identity') else ''})")
    if res["issues"]:
        lines += ["", "**I also detected:**"] + [f"- {i['message']}" for i in res["issues"]]
    if len(res["components"]) > 1:
        lines += ["", f"The datasets form {len(res['components'])} separate integration groups: " + "; ".join(", ".join(c) for c in res["components"])]
    return "\n".join(lines)


def r_plan(plan: dict[str, Any]) -> str:
    t = plan["thresholds"]
    lines = [f"**Integration plan** (`{plan['plan_id']}`, mode **{plan['mode']}**, conflicts: **{plan['conflict_strategy']}**)", "",
             f"Grain: {plan['grain']}. Thresholds: auto-merge ≥ {t['auto_merge']:.2f}, review ≥ {t['review']:.2f}, relationships ≥ {t['min_edge']:.2f}.", ""]
    for s in plan["steps"]:
        flag = " ⚠️ review recommended" if s.get("requires_review") else ""
        lines.append(f"{s['step']}. {s['description']}{flag}")
    if plan["excluded_relationships"]:
        lines += ["", "Relationships not used:"] + [f"- {e['left']} ↔ {e['right']} ({e['confidence']:.2f}): {e['reason']}" for e in plan["excluded_relationships"]]
    if plan["warnings"]:
        lines += ["", "Notes:"] + [f"- {w}" for w in plan["warnings"]]
    cols = [c["name"] for c in plan["canonical_schema"] if not c["name"].startswith("_")]
    lines += ["", f"Unified schema ({len(plan['canonical_schema'])} columns): " + ", ".join(cols[:40]) + (" …" if len(cols) > 40 else "")]
    return "\n".join(lines)


def r_merge(res: dict[str, Any]) -> str:
    q = res["quality_report"]
    lines = [f"Merge `{res['merge_id']}` completed in {res['elapsed_ms'] / 1000:.1f}s: **{res['row_count']:,} rows × {res['column_count']} columns**.", "", "```", q["text"], "```"]
    failed = [c for c in res["validation"]["checks"] if not c["passed"]]
    if failed:
        lines += ["", "Validation findings:"] + [f"- {c['check']}: {c['details']}" for c in failed]
    lines.append("\nYou can ask me to show conflicts, duplicates, provenance, or export the result.")
    return "\n".join(lines)


def r_conflicts(res: dict[str, Any]) -> str:
    if not res["total"]:
        return "There are no conflicting values in the current merge — every fused attribute agreed across sources."
    lines = [f"**{res['total']} conflicts** (strategy: {res['strategy']}). By attribute: " + ", ".join(f"{k}: {v}" for k, v in res["by_attribute"].items()), ""]
    for c in res["conflicts"][:10]:
        cands = "; ".join(f"'{x['value']}' ({x['source_dataset']}.{x['source_column']} row {x['source_row']}{' ✔' if x['selected'] else ''})" for x in c["candidates"])
        lines.append(f"- `{c['entity_id']}` **{c['attribute']}** → '{c['resolved_value']}' — {c['reason']}. Candidates: {cands}")
    if res["matching"] > 10:
        lines.append(f"\n…and {res['matching'] - 10} more (see conflicts.csv).")
    return "\n".join(lines)


def r_duplicates(res: dict[str, Any]) -> str:
    lines = []
    if res["exact_duplicate_rows"]:
        lines.append("Exact duplicate rows: " + ", ".join(f"{d}: {n}" for d, n in res["exact_duplicate_rows"].items()))
    for g in res["entity_groups"]:
        lines.append(f"Entity group {g['group']} ({', '.join(g['datasets'])}): **{g['within_dataset_duplicates_resolved']} duplicate records** resolved within datasets, {g['cross_source_links']} cross-source links.")
        for p in g["examples"][:8]:
            fields = ", ".join(f"{k}={v}" for k, v in p["field_levels"].items())
            lines.append(f"  - {p['left_dataset']} row {p['left_row']} ≡ row {p['right_row']} (p={p['probability']:.3f}; {fields})")
    return "\n".join(lines) or "No duplicates were found."


def r_explain(res: dict[str, Any]) -> str:
    if "text" in res:
        return f"```\n{res['text']}\n```"
    if "summary" in res and "key_matches" in res:
        lines = [f"**{res['datasets'][0]} ↔ {res['datasets'][1]}** — {res['join_kind'].replace('_', ' ')}, confidence {res['confidence']:.2f} ({res['band']})", "", res["summary"], ""]
        for km in res["key_matches"]:
            lines.append(f"```\n{km['text']}\n```")
        return "\n".join(lines)
    return res.get("summary") or res.get("explanation") or json.dumps(res, default=str)[:1500]


def r_route(res: dict[str, Any]) -> str:
    return res.get("explanation") or ("No route found." if not res.get("found") else json.dumps(res)[:800])


def r_provenance(res: dict[str, Any]) -> str:
    if "column" in res:
        srcs = ", ".join(f"{s['dataset']}.{s['column']}" + (f" [{s['transformation']}]" if s.get("transformation") else "") for s in res["sources"])
        steps = "\n".join(f"  {i + 1}. {h['operation']}" + (f" via {h['join_keys']}" if h.get("join_keys") else "") + (f" (role {h['role']})" if h.get("role") else "") + (f", confidence {h['confidence']:.2f}" if h.get("confidence") else "") + (f" — strategy {h['strategy']}" if h.get("strategy") else "") for i, h in enumerate(res["path"]))
        return f"**unified.{res['column']}** ← {srcs}\n{steps}"
    lines = ["**Column lineage**", ""]
    for col, info in list(res["columns"].items())[:60]:
        if col.startswith("_"):
            continue
        lines.append(f"- unified.{col} ← " + ", ".join(f"{s['dataset']}.{s['column']}" for s in info["sources"]))
    lines.append(f"\nRow lineage columns: {', '.join(res['row_lineage_columns'])}")
    return "\n".join(lines)


def r_text(res: Any) -> str:
    return res if isinstance(res, str) else f"```\n{json.dumps(res, indent=2, default=str)[:3000]}\n```"


def r_stats(res: dict[str, Any]) -> str:
    lines = [f"**{res['match_percentage']:.1f}%** of the {res['rows']:,} unified rows were matched in every lookup ({res['matched_rows']:,} matched, {res['unmatched_rows']:,} unmatched)."]
    for j in res["joins"]:
        lines.append(f"- {j['parent']} ← {j['child']}{' as ' + j['role'] if j['role'] else ''}: {_pct(j['match_rate'])}")
    if res["entity_records"]:
        lines.append(f"- Entity resolution: {res['linked_entity_records']:,} of {res['entity_records']:,} records linked to at least one other record ({res['entities']:,} entities)")
    lines.append(f"Overall integration coverage: {_pct(res['integration_coverage'])}; conflicts: {res['conflicts']}")
    return "\n".join(lines)


def r_mappings(res: dict[str, Any]) -> str:
    lines = ["**Inferred column mappings**", "", "| left | right | confidence | band | status | values via |", "|---|---|---|---|---|---|"]
    for m in res["mappings"]:
        lines.append(f"| {m['left']} | {m['right']} | {m['confidence']:.2f} | {m['band']} | {m['status']} | {m['normalizer']} |")
    if res.get("rejected_candidates"):
        lines += ["", "Top rejected candidates:"] + [f"- {m['left']} ↔ {m['right']} ({m['confidence']:.2f}): {m['rejection_reason']}" for m in res["rejected_candidates"][:8]]
    return "\n".join(lines)


def r_entities(res: dict[str, Any]) -> str:
    st = res["stats"]
    lines = [f"Entity matching across {', '.join(res['datasets'])}: {st['records']:,} records, {st['candidate_pairs']:,} candidate pairs after blocking "
             f"({st['reduction_ratio']:.2%} of {st['possible_pairs']:,} possible pairs avoided).",
             f"**{st['auto_links']} confident links**, **{st['review_links']} ambiguous links** needing review → {st['entities']:,} entities ({res['clusters_with_multiple_records']} with multiple records).", ""]
    for p in res["uncertain_pairs"][:5]:
        lines.append(f"- ambiguous: {p['left_dataset']}#{p['left_row']} vs {p['right_dataset']}#{p['right_row']} p={p['probability']:.2f} ({', '.join(f'{k}:{v}' for k, v in p['field_levels'].items())})")
    return "\n".join(lines)


def r_prefs(res: dict[str, Any]) -> str:
    t = res["thresholds"]
    extra = f", only matches ≥ {res['confidence_threshold']:.0%}" if res.get("confidence_threshold") else ""
    pk = f", primary identifier **{res['primary_key']}**" if res.get("primary_key") else ""
    unc = ", uncertain records will **not** be merged" if res.get("merge_uncertain") is False else ""
    return f"Preferences updated: mode **{res['mode']}**, conflict strategy **{res['conflict_strategy']}**{extra}{pk}{unc} (mode thresholds: auto {t['auto_merge']}, review {t['review']})."


def r_export(res: dict[str, Any]) -> str:
    return "Exported files:\n" + "\n".join(f"- `{name}`" for name in res["files"]) + "\n\nDownload them from the Output tab or `GET /integration/export?file=<name>`."


def r_graph(res: dict[str, Any]) -> str:
    st = res["stats"]
    lines = [f"The integration graph has {st['nodes']} nodes and {st['edges']} edges ({', '.join(f'{k}: {v}' for k, v in st['edge_types'].items())}). Open the **Graph** tab to explore it; click an edge for evidence.", ""]
    for e in res["edges"][:12]:
        lines.append(f"- {e['source']} — {e['target']}: {e['label']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- runners
def _num(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    v = args.get(key, default)
    try:
        v = int(v)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError(f"{key} must be an integer") from exc
    return max(lo, min(hi, v))


def _profile(s: Session, a: dict[str, Any]) -> dict[str, Any]:
    ds = s.resolve_dataset_id(a["dataset"])
    prof = s.profiles.get(ds) or s.profile(ds)
    return {"dataset_id": ds, "source_name": s.artifacts[ds].source_name, "profile": prof.model_dump(mode="json")}


def _plan(s: Session, a: dict[str, Any]) -> dict[str, Any]:
    if any(k in a for k in ("confidence_threshold", "primary_key", "merge_uncertain")):
        s.set_preferences(**{k: a[k] for k in ("confidence_threshold", "primary_key", "merge_uncertain") if k in a})
    return s.make_plan(a.get("mode"), a.get("conflict_strategy"), a.get("datasets")).model_dump(mode="json")


def _execute(s: Session, a: dict[str, Any]) -> dict[str, Any]:
    if any(k in a for k in ("mode", "conflict_strategy", "confidence_threshold", "primary_key", "merge_uncertain")):
        s.set_preferences(**{k: a[k] for k in ("mode", "conflict_strategy", "confidence_threshold", "primary_key", "merge_uncertain") if k in a})
    rec = s.execute()
    out = rec.result.model_dump(mode="json")
    out["plan"] = rec.plan.model_dump(mode="json")
    out["files"] = sorted(rec.files)
    return out


TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    TOOLS[tool.name] = tool


register(Tool("list_datasets", "List the datasets loaded in this session with their size and format.", _obj(), lambda s, a: s.list_datasets(), r_datasets, "datasets"))
register(Tool("inspect_dataset", "Show a preview of rows of one dataset.", _obj({"dataset": ds_param, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ["dataset"]),
              lambda s, a: s.preview(a["dataset"], _num(a, "limit", 10, 1, 100)), r_text, "table"))
register(Tool("profile_dataset", "Profile a dataset: types, semantic types, nulls, cardinality, statistics, duplicates.", _obj({"dataset": ds_param}, ["dataset"]), _profile, r_profile, "profile"))
register(Tool("compare_schemas", "Compare the columns of two datasets and return every scored correspondence with its evidence.", _obj({"left": ds_param, "right": ds_param}, ["left", "right"]),
              lambda s, a: s.compare_schemas(a["left"], a["right"]), lambda r: "\n\n".join(f"```\n{m['text']}\n```" for m in r["matches"][:8]) or "No scored correspondences.", "mappings"))
register(Tool("find_candidate_relationships", "Inspect all loaded datasets: profile schemas, match columns, build the integration graph and report relationships, mappings and data issues.",
              _obj({"force": {"type": "boolean"}}), lambda s, a: s.discover(force=bool(a.get("force"))), r_discovery, "discovery"))
register(Tool("show_mappings", "List the column mappings inferred between datasets (accepted and top rejected).", _obj(), lambda s, a: s.discover(), r_mappings, "mappings"))
register(Tool("build_graph", "Return the integration graph of datasets and relationships (level 'dataset' or 'column').", _obj({"level": {"type": "string", "enum": ["dataset", "column"]}}),
              lambda s, a: s.graph_view(a.get("level", "dataset")), r_graph, "graph"))
register(Tool("find_entity_matches", "Run entity matching (record linkage) between two datasets, or deduplication of one, without merging. Reports confident and ambiguous matches.",
              _obj({"left": ds_param, "right": ds_param}, ["left"]), lambda s, a: s.entity_match_preview(a["left"], a.get("right")), r_entities, "entities"))
register(Tool("set_preferences", "Set merge preferences: mode (strict/balanced/permissive), conflict strategy, minimum confidence (0-1 or percent), primary identifier column, whether uncertain records may be merged, and source priority.",
              _obj({"mode": {"type": "string", "enum": list(MODES)}, "conflict_strategy": {"type": "string", "enum": list(CONFLICT_STRATEGIES)},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100}, "primary_key": S, "merge_uncertain": {"type": "boolean"},
                    "source_priority": {"type": "array", "items": S}}),
              lambda s, a: s.set_preferences(**a), r_prefs, "preferences", mutating=True))
register(Tool("generate_merge_plan", "Generate the merge plan (order, join keys, join types, transformations, conflict rules, thresholds) without executing it.",
              _obj({"mode": {"type": "string", "enum": list(MODES)}, "conflict_strategy": {"type": "string", "enum": list(CONFLICT_STRATEGIES)},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100}, "primary_key": S, "merge_uncertain": {"type": "boolean"},
                    "datasets": {"type": "array", "items": S}}),
              _plan, r_plan, "plan", mutating=True))
register(Tool("execute_merge", "Execute the integration (plans first if needed), validate it, and write all outputs.",
              _obj({"mode": {"type": "string", "enum": list(MODES)}, "conflict_strategy": {"type": "string", "enum": list(CONFLICT_STRATEGIES)},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 100}, "primary_key": S, "merge_uncertain": {"type": "boolean"}}),
              _execute, r_merge, "merge", mutating=True))
register(Tool("show_conflicts", "Show conflicting values found while fusing entities, with all candidate values and their sources.",
              _obj({"attribute": S, "entity_id": S, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}),
              lambda s, a: s.conflicts(_num(a, "limit", 20, 1, 200), a.get("attribute"), a.get("entity_id")), r_conflicts, "conflicts"))
register(Tool("show_duplicates", "Show duplicate entities that were resolved (within-dataset duplicates and cross-source links).", _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 100}}),
              lambda s, a: s.duplicates(_num(a, "limit", 10, 1, 100)), r_duplicates, "duplicates"))
register(Tool("explain_match", "Explain why two columns were (or were not) matched, with every signal, weight, value normaliser and graph adjustment. Columns as 'dataset.column' or bare column names.",
              _obj({"left": S, "right": S}, ["left", "right"]), lambda s, a: s.explain_match(a["left"], a["right"]), r_explain, "explanation"))
register(Tool("explain_relationship", "Explain why two datasets were linked (join kind, keys, coverage, evidence), or the route connecting them.",
              _obj({"left": ds_param, "right": ds_param}, ["left", "right"]), lambda s, a: s.explain_relationship(a["left"], a["right"]), r_explain, "explanation"))
register(Tool("find_integration_route", "Find the most reliable integration route between two datasets with Dijkstra (cost = −log confidence), compared with the direct relationship.",
              _obj({"source": ds_param, "target": ds_param, "cost_mode": {"type": "string", "enum": ["neglog", "linear"]}}, ["source", "target"]),
              lambda s, a: s.route(a["source"], a["target"], a.get("cost_mode")), r_route, "route"))
register(Tool("decide_mapping", "Approve or reject a column mapping (e.g. when the user says a match is wrong).",
              _obj({"left": S, "right": S, "decision": {"type": "string", "enum": ["approved", "rejected"]}}, ["left", "right", "decision"]),
              lambda s, a: s.decide_match(a["left"], a["right"], a["decision"]), lambda r: f"Mapping {r['left']} ↔ {r['right']} is now **{r['status']}**. The plan will be regenerated on the next merge.", "mappings", mutating=True))
def r_review(res: dict[str, Any]) -> str:
    if not res["pairs"]:
        return "No uncertain record pairs left to review."
    lines = [f"Record pairs worth labelling ({res['labels']} labelled so far):"]
    for q in res["pairs"][:10]:
        vals = " vs ".join("; ".join(str(v) for v in q[s]["values"].values() if v is not None) for s in ("left", "right"))
        lines.append(f"- {q['left']['source']} row {q['left']['row']} ↔ {q['right']['source']} row {q['right']['row']}: p={q['probability']:.2f} [{q['sampling']}] ({vals})")
    return "\n".join(lines)


register(Tool("review_entity_links", "List record pairs from the last merge that are most useful for the user to label (active learning).",
              _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 50}}), lambda s, a: s.entity_review_queue(_num(a, "limit", 10, 1, 50)), r_review, "review"))
register(Tool("label_entity_link", "Label two records as the same entity (match) or not (non_match); applied on the next merge.",
              _obj({"left_dataset": ds_param, "left_row": {"type": "integer", "minimum": 0}, "right_dataset": ds_param, "right_row": {"type": "integer", "minimum": 0},
                    "decision": {"type": "string", "enum": ["match", "non_match"]}}, ["left_dataset", "left_row", "right_dataset", "right_row", "decision"]),
              lambda s, a: s.label_entity_link({"dataset": a["left_dataset"], "row": a["left_row"]}, {"dataset": a["right_dataset"], "row": a["right_row"]}, a["decision"]),
              lambda r: f"Labelled as **{r['decision']}** ({r['labels']} labels in total). Re-run the merge to apply.", "review", mutating=True))
def r_uncertain_agg(res: dict[str, Any]) -> str:
    what = "rows" if res["agg"] == "count" else f"{res['agg']}({res['measure']})"
    lines = [f"{what} by {res['group_by'] or 'all rows'} with linkage uncertainty ({res['rows_below_threshold']} of {res['rows']} rows below the {res['certain_threshold']} threshold; expected correctly integrated rows {res['expected_correct_rows']}):",
             "| group | point | expected | 95% interval | certain-only |", "|---|---|---|---|---|"]
    for g in res["groups"][:15]:
        ci = f"{g['ci95_low']} – {g['ci95_high']}" if g["ci95_low"] is not None else "–"
        lines.append(f"| {g['group']} | {g['point']} | {g['expected']} | {ci} | {g['certain_only']} |")
    for sc in res.get("scenarios") or []:
        first = next(iter(sc["if_wrong"].items()), None)
        lines.append(f"\nIf the relationship {sc['relationship']} is wrong (probability {sc['probability_wrong']:.0%}), rows joined through it drop out"
                     + (f": e.g. {first[0]} → {first[1]}." if first else "."))
    return "\n".join(lines)


def r_history(res: dict[str, Any]) -> str:
    if not res["rows"]:
        return res.get("note") or "No history rows match."
    lines = [f"{res['total']} history rows ({res.get('attributes_with_changes', 0)} entity attributes changed over time)" + (f", valid at {res['as_of']}" if res.get("as_of") else "") + ":"]
    for h in res["rows"][:15]:
        span = f"{(h['valid_from'] or '?')[:10]} → {(h['valid_to'] or 'now')[:10]}" if h["valid_from"] else "undated"
        lines.append(f"- {h['entity_id']} {h['attribute']} = {h['value']} ({span}, {h['timestamp_kind']}, from {h['source_dataset']} row {h['source_row']})")
    return "\n".join(lines)


register(Tool("aggregate_with_uncertainty", "Aggregate the unified dataset (sum/count/avg of a column, optionally grouped) and show how much of each figure depends on uncertain matches.",
              _obj({"measure": S, "group_by": S, "agg": {"type": "string", "enum": ["sum", "count", "avg"]}}),
              lambda s, a: s.aggregate_with_uncertainty(a.get("measure"), a.get("group_by"), a.get("agg") or "sum", 30), r_uncertain_agg, "table"))
register(Tool("entity_history", "Show how entity attributes changed over time (validity intervals), optionally for one entity/attribute or as of a date.",
              _obj({"entity_id": S, "attribute": S, "as_of": {"type": "string", "description": "ISO date"}, "as_known_at": {"type": "string", "description": "ISO date: as recorded by the merge at that time"}}),
              lambda s, a: s.entity_history(a.get("entity_id"), a.get("attribute"), a.get("as_of"), 100, a.get("as_known_at")), r_history, "history"))
register(Tool("show_provenance", "Show lineage of the unified dataset, or of one output column.", _obj({"column": S}),
              lambda s, a: s.provenance(a.get("column")), r_provenance, "provenance"))
register(Tool("lineage_report", "Generate a human-readable data lineage report for the current merge.", _obj(), lambda s, a: s.lineage_report(), lambda r: f"```\n{r[:6000]}\n```", "provenance"))
register(Tool("match_statistics", "Report what percentage of rows and records were matched in the current merge.", _obj(), lambda s, a: s.stats(), r_stats, "stats"))
register(Tool("quality_report", "Show the data quality report of the current merge.", _obj(), lambda s, a: s.current_merge().result.quality_report, lambda r: f"```\n{r['text']}\n```", "quality"))
register(Tool("preview_output", "Preview rows of the unified dataset.", _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 100}, "columns": {"type": "array", "items": S}}),
              lambda s, a: s.output_preview(_num(a, "limit", 10, 1, 100), columns=a.get("columns")), r_text, "table"))
register(Tool("undo_merge", "Undo the most recent merge (outputs are archived; sources are never modified).", _obj(), lambda s, a: s.undo(),
              lambda r: f"Undid merge `{r['undone']}`. Its outputs were archived to `{r['archived_to']}` (nothing deleted; sources untouched). "
              + (f"The current merge is now `{r['current_merge']}`." if r["current_merge"] else "There is no active merge now."), None, mutating=True))
register(Tool("export_dataset", "List the exported files of the current merge (unified dataset, reports, provenance, conflicts, mappings, graph).", _obj(),
              lambda s, a: {"files": s.export_paths()}, r_export, "export"))


def validate_arguments(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Validation against the tool schema: types, enums, ranges.

    Unknown argument names are dropped (and logged) rather than failing the call: models sometimes add
    parameters that do not exist (``match_statistics(dataset=...)``). Dropping them is safe because only
    schema-declared arguments ever reach the session method; wrong *types or values* still fail."""
    if not isinstance(args, dict):
        raise ToolArgumentError(f"{tool.name}: arguments must be an object")
    if "__unparseable__" in args:
        raise ToolArgumentError(f"{tool.name}: arguments were not valid JSON")
    props = tool.parameters["properties"]
    clean: dict[str, Any] = {}
    for k, v in args.items():
        if v is None:
            continue
        if k not in props:
            log.info("Dropped unknown tool argument", tool=tool.name, argument=k)
            continue
        spec = props[k]
        t = spec.get("type")
        ok = {
            "string": isinstance(v, str),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": isinstance(v, bool),
            "array": isinstance(v, list),
        }.get(t, True)
        if t == "integer" and isinstance(v, str) and v.isdigit():
            v, ok = int(v), True
        if t == "number" and isinstance(v, str):
            try:
                v, ok = float(v.rstrip("%")), True
            except ValueError:
                ok = False
        if t == "boolean" and isinstance(v, str) and v.lower() in ("true", "false"):
            v, ok = v.lower() == "true", True
        if not ok:
            raise ToolArgumentError(f"{tool.name}: argument {k!r} must be {t}")
        if "enum" in spec and v not in spec["enum"]:
            raise ToolArgumentError(f"{tool.name}: argument {k!r} must be one of {spec['enum']}")
        if isinstance(v, str) and len(v) > 500:
            raise ToolArgumentError(f"{tool.name}: argument {k!r} is too long")
        clean[k] = v
    for req in tool.parameters.get("required", []):
        if req not in clean:
            raise ToolArgumentError(f"{tool.name}: missing required argument {req!r}")
    return clean


def execute_tool(session: Session, name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "tool": name, "error": f"Unknown tool {name!r}", "rendered": f"I can't do '{name}' — it is not an available operation."}
    try:
        clean = validate_arguments(tool, args)
        result = tool.run(session, clean)
        return {"ok": True, "tool": name, "arguments": clean, "result": result, "rendered": tool.render(result), "card": tool.card}
    except DFGError as exc:
        return {"ok": False, "tool": name, "arguments": args, "error": exc.message, "details": exc.details, "rendered": f"⚠️ {exc.message}"}


def compact_for_llm(res: dict[str, Any], limit: int = 6000) -> str:
    """What the model sees of a tool result: the rendered text (already factual and bounded)."""
    if not res["ok"]:
        return json.dumps({"ok": False, "error": res["error"]})
    return res["rendered"][:limit]
