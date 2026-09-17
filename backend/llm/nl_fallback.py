"""Deterministic natural-language → tool-call parser.

Used by the mock provider (no API key needed), as the automatic fallback when
a remote LLM is unavailable, and in tests. It recognises the documented
conversational commands, extracts dataset / column mentions from session
state, and composes multi-intent requests ("only merge matches above 95%
and prefer the latest values, then merge") into ordered tool calls.
"""
from __future__ import annotations

import re
from typing import Any

from backend.session.state import Session

_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%|(?:above|over|at least|>=?|threshold(?: of)?)\s*(0?\.\d+|\d{1,3}(?:\.\d+)?)")


def _mentions_datasets(session: Session, text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    low = text.lower()
    for ds, art in session.artifacts.items():
        names = {ds.lower(), art.source_name.lower(), art.source_name.lower().rsplit(".", 1)[0]}
        pos = min((low.find(n) for n in names if n and low.find(n) >= 0), default=-1)
        if pos >= 0:
            found.append((pos, ds))
    return [d for _, d in sorted(found)]


def _mentions_columns(session: Session, text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for ds, prof in session.profiles.items():
        for c in prof.columns:
            for m in re.finditer(r"(?<![\w.])" + re.escape(c.name) + r"(?![\w])", text, flags=re.IGNORECASE):
                # qualify when written as dataset.column
                prefix = text[max(0, m.start() - len(ds) - 1): m.start()]
                name = f"{ds}.{c.name}" if prefix.lower() == f"{ds.lower()}." else c.name
                found.append((m.start(), name))
    seen, out = set(), []
    for _, n in sorted(found):
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    if session.merges:
        try:
            for col in session.current_merge().lineage:
                if re.search(r"(?<![\w.])" + re.escape(col) + r"(?![\w])", text, flags=re.IGNORECASE) and col.lower() not in seen:
                    out.append(col)
                    seen.add(col.lower())
        except Exception:  # noqa: BLE001 - no active merge
            pass
    return out


def _threshold(text: str) -> float | None:
    if not re.search(r"\b(only|above|over|at least|threshold|confidence|minimum|min)\b", text):
        return None
    m = _PCT.search(text)
    if not m:
        return None
    v = float(m.group(1) or m.group(2))
    return v / 100 if v > 1 else v


def parse(session: Session, message: str) -> list[tuple[str, dict[str, Any]]]:
    text = message.strip()
    low = text.lower()
    calls: list[tuple[str, dict[str, Any]]] = []
    datasets = _mentions_datasets(session, text)
    columns = _mentions_columns(session, text)
    has_merge = any(r.status == "active" for r in session.merges)

    # ---------------------------------------------------------------- preferences
    prefs: dict[str, Any] = {}
    consumed: list[str] = []  # preference phrases are removed before action detection
    thr = _threshold(low)
    if thr is not None:
        prefs["confidence_threshold"] = thr
    for mode in ("strict", "balanced", "permissive"):
        if re.search(rf"\b{mode}\b", low):
            prefs["mode"] = mode
    unc = re.search(r"(don'?t|do not|never|avoid)\s+(auto-?)?merg\w*\s+(any\s+)?(uncertain|ambiguous|low[- ]confidence)\s*\w*", low)
    if unc or "only certain" in low:
        prefs["merge_uncertain"] = False
        if unc:
            consumed.append(unc.group(0))
    m = re.search(r"use\s+([\w.]+)\s+as\s+(the\s+)?(primary|main)?\s*(identifier|key|id)", low)
    if m:
        prefs["primary_key"] = m.group(1)
        consumed.append(m.group(0))
    strategies = {
        r"prefer(ring)?\s+(the\s+)?(latest|newest|most recent)": "prefer_latest",
        r"majority": "majority_vote",
        r"prefer(ring)?\s+non[- ]?null": "prefer_non_null",
        r"highest confidence": "highest_confidence",
        r"keep all": "keep_all",
        r"manual review|let me review": "manual_review",
        r"source accuracy|truth discovery|most reliable source": "source_accuracy_vote",
        r"prefer(ring)?\s+(the\s+)?source|trust\s+\w+\s+more": "prefer_source",
    }
    for pat, strat in strategies.items():
        if re.search(pat, low):
            prefs["conflict_strategy"] = strat
            break
    if prefs:
        calls.append(("set_preferences", prefs))
    for phrase in consumed:
        low = low.replace(phrase, " ")

    # ---------------------------------------------------------------- actions
    def add(name: str, args: dict[str, Any] | None = None) -> None:
        calls.append((name, args or {}))

    is_question_why = bool(re.search(r"\bwhy\b|\bexplain\b|\bhow come\b|\breason\b", low))
    if re.search(r"\bundo\b|\brevert\b|\broll ?back\b", low):
        add("undo_merge")
    elif re.search(r"lineage report|provenance report|data lineage", low):
        add("lineage_report")
    elif re.search(r"\bexport\b|\bdownload\b|\bsave\b", low):
        add("export_dataset")
    elif re.search(r"\bprovenance\b|\blineage\b|where does .* come from|comes from", low):
        add("show_provenance", {"column": columns[0].split(".")[-1]} if columns else {})
    elif is_question_why and re.search(r"\bjoin\b", low) and not datasets:
        plan = session.plan or (session.current_merge().plan if has_merge else None)
        if plan and plan.integration_tree:
            e = next((x for x in plan.integration_tree if x["join_kind"] in ("lookup", "aggregate_lookup", "reverse_aggregate")), plan.integration_tree[0])
            add("explain_relationship", {"left": e["parent"], "right": e["child"]})
        elif session.relationships:
            r = session.relationships[0]
            add("explain_relationship", {"left": r.left_dataset, "right": r.right_dataset})
        else:
            add("find_candidate_relationships")
    elif is_question_why and len(datasets) >= 2 and not (len(columns) >= 2 and re.search(r"column|match|map", low)):
        add("explain_relationship", {"left": datasets[0], "right": datasets[1]})
    elif is_question_why and len(columns) >= 2:
        add("explain_match", {"left": columns[0], "right": columns[1]})
    elif re.search(r"\broute\b|\bpath\b|how (is|are) .* (connected|related|linked)", low) and len(datasets) >= 2:
        add("find_integration_route", {"source": datasets[0], "target": datasets[1]})
    elif re.search(r"\bconflict", low):
        args: dict[str, Any] = {}
        if columns:
            args["attribute"] = columns[0].split(".")[-1]
        add("show_conflicts", args)
    elif re.search(r"\bduplicat", low):
        if has_merge:
            add("show_duplicates")
        elif datasets:
            add("find_entity_matches", {"left": datasets[0]})
        elif session.artifacts:
            add("find_entity_matches", {"left": next(iter(session.artifacts))})
    elif re.search(r"percent(age)?|match rate|how many .*match|matched rows|what share", low):
        add("match_statistics")
    elif re.search(r"quality report|data quality", low):
        add("quality_report")
    elif re.search(r"\bgraph\b|visuali[sz]e", low):
        add("build_graph", {"level": "column" if "column" in low else "dataset"})
    elif re.search(r"columns?\s+(refer|correspond|map)|which columns|same attribute", low):
        add("show_mappings")
    elif re.search(r"(same|matching|match)\s+entit|record linkage|entity (resolution|matching)|ambiguous", low):
        if len(datasets) >= 2:
            pair = datasets[:2]
        else:
            er = [r for r in session.relationships if r.join_kind.value in ("entity_resolution", "entity_key_merge")]
            pair = [er[0].left_dataset, er[0].right_dataset] if er else list(session.artifacts)[:2]
        if pair:
            add("find_entity_matches", {"left": pair[0], **({"right": pair[1]} if len(pair) > 1 else {})})
    elif re.search(r"(reject|wrong|incorrect|not (a )?match)", low) and len(columns) >= 2:
        add("decide_mapping", {"left": columns[0], "right": columns[1], "decision": "rejected"})
    elif re.search(r"\b(approve|confirm|correct match)\b", low) and len(columns) >= 2:
        add("decide_mapping", {"left": columns[0], "right": columns[1], "decision": "approved"})
    elif re.search(r"(what|which|show).*(mapping|correspond)|columns refer to the same|inferred", low):
        add("show_mappings")
    elif re.search(r"compare", low) and len(datasets) >= 2:
        add("compare_schemas", {"left": datasets[0], "right": datasets[1]})
    elif re.search(r"\bplan\b|how (would|will) you merge|what would you do", low):
        add("generate_merge_plan")
    elif re.search(r"\b(merge|integrate|combine|unify|join them|go ahead|proceed|do it)\b", low) and not (prefs and re.search(r"only merge|merge only", low)):
        add("execute_merge")
    elif re.search(r"preview|show (me )?(the )?(result|output|unified)", low) and has_merge:
        add("preview_output")
    elif re.search(r"schema|profile|column types|semantic type", low):
        for ds in (datasets or list(session.artifacts)):
            add("profile_dataset", {"dataset": ds})
    elif re.search(r"(list|what|which).*datasets|what.*loaded", low):
        add("list_datasets")
    elif re.search(r"(show|preview|inspect|look at)\b", low) and datasets:
        add("inspect_dataset", {"dataset": datasets[0], "limit": 10})
    elif re.search(r"relationship|relate|inspect|analy[sz]e|discover|find|load|connect", low) and len(session.artifacts) >= 2:
        add("find_candidate_relationships")
    elif re.search(r"\b(load|upload)\b", low):
        add("list_datasets")

    if not calls or all(c[0] == "set_preferences" for c in calls):
        if prefs and re.search(r"\b(merge|integrate|combine)\b", low) and not re.search(r"only merge|merge only", low):
            add("execute_merge")
        elif not calls:
            if len(session.artifacts) >= 2 and session.discovery_stale:
                add("find_candidate_relationships")
            elif not session.artifacts:
                add("list_datasets")
    return calls


HELP = """I can help you integrate datasets. Try:
- "Find relationships between the datasets" / "What mappings did you infer?"
- "Show me the merge plan" / "Merge everything" / "Only merge matches above 95%"
- "Use customer_id as the primary identifier" / "Don't merge uncertain records" / "Prefer the latest values"
- "Show me conflicts" / "Show me duplicate entities" / "What percentage of rows were matched?"
- "Explain why customers and sales were linked" / "Why do city and location match?" / "Why did you choose this join?"
- "Show me the graph" / "Generate a data lineage report" / "Export the final dataset" / "Undo the previous merge\""""
