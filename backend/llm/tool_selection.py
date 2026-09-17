"""Relevance-based tool selection: send only the tool schemas a request can need.

The 28 tool schemas are ~2.5K of the ~3.5K prompt tokens of every agent round,
and Groq's free tier allows 8K tokens per minute. Sending a relevant subset
roughly halves the cost of a turn. Selection is deterministic and cheap:

    selected = CORE(state)                                   always-useful actions for the session state
             ∪ tools named by the rule-based parser          (its intent grammar is precise when it fires)
             ∪ top-scoring tools by keyword relevance       until ``max_tools``

Keyword relevance scores a tool by overlap between the message tokens and the
tool's name, description and a small synonym list (e.g. "revert", "scrap" →
undo_merge). Selection recall on the 40 labelled benchmark requests is checked
by ``tests/test_llm_agent.py`` without calling any LLM.
"""
from __future__ import annotations

import re
from typing import Any

from backend.llm.tools import TOOLS

SYNONYMS: dict[str, str] = {
    "list_datasets": "datasets loaded files tables which what have",
    "inspect_dataset": "preview rows sample look show first records inside",
    "profile_dataset": "profile schema schemas columns types statistics nulls inspect describe summary",
    "compare_schemas": "compare schemas columns same correspond differences between",
    "find_candidate_relationships": "relationships relate related connect connected link linked discover inspect how together schemas analyse analyze figure",
    "show_mappings": "mappings mapping columns same entity correspond refer inferred matched columns",
    "build_graph": "graph network diagram visualise visualize relationships",
    "find_entity_matches": "entity matching duplicates same person customers records link records",
    "set_preferences": "only above below percent threshold confidence trust sure conservative strict permissive prefer latest recent newest primary identifier key uncertain priority keep value disagree strategy mode",
    "generate_merge_plan": "plan steps order how would combine walk through before preview approach",
    "execute_merge": "merge combine join integrate unify go ahead run everything one table",
    "show_conflicts": "conflicts conflicting disagree disagreement inconsistent differ contradict details",
    "show_duplicates": "duplicates duplicate dedup deduplicate repeated same entities",
    "explain_match": "why explain column matched mapping reason evidence convinced sure know same thing",
    "explain_relationship": "why explain join linked relationship datasets chose choose reason",
    "find_integration_route": "route path connected hops through via reach",
    "decide_mapping": "wrong incorrect approve reject not match mapping correct",
    "show_provenance": "provenance lineage where come from origin source derived column",
    "lineage_report": "lineage report audit trail",
    "match_statistics": "percentage percent how many matched statistics rate coverage ended up",
    "quality_report": "quality report issues validation",
    "preview_output": "output result unified final rows show preview look",
    "undo_merge": "undo revert rollback scrap start over previous last",
    "export_dataset": "export download save file files final",
    "review_entity_links": "review label uncertain pairs unsure check ambiguous records",
    "label_entity_link": "label same entity different not same match non_match records rows",
    "entity_history": "history changed over time previous earlier moved valid as of timeline",
    "aggregate_with_uncertainty": "total sum count average aggregate uncertainty confidence interval how much figures expected",
}

_STOP = {"the", "a", "an", "of", "to", "and", "or", "is", "are", "me", "my", "i", "you", "it", "this", "that", "these", "those", "in", "on", "for", "with", "be", "do", "can", "please", "show", "what", "which"}


def _tokens(text: str) -> set[str]:
    toks = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        toks.add(t)
        if t.endswith("s") and len(t) > 3:
            toks.add(t[:-1])
        if t.endswith("ing") and len(t) > 5:
            toks.add(t[:-3])
        if t.endswith("ed") and len(t) > 4:
            toks.add(t[:-2])
    return toks


_TOOL_TOKENS = {name: _tokens(name.replace("_", " ") + " " + t.description + " " + SYNONYMS.get(name, "")) for name, t in TOOLS.items()}


def core_tools(session: Any) -> list[str]:
    if not getattr(session, "artifacts", None):
        return ["list_datasets"]
    has_merge = any(r.status == "active" for r in session.merges)
    core = ["find_candidate_relationships", "set_preferences", "execute_merge", "generate_merge_plan"]
    if has_merge:
        core += ["show_conflicts", "match_statistics", "undo_merge"]
    return core


def select_tools(session: Any, message: str, max_tools: int = 12) -> list[str]:
    """Names of the tools to offer the model for this request (all tools when ``max_tools`` ≤ 0)."""
    if max_tools <= 0 or max_tools >= len(TOOLS):
        return list(TOOLS)
    chosen: list[str] = []

    def add(name: str) -> None:
        if name in TOOLS and name not in chosen:
            chosen.append(name)

    for name in core_tools(session):
        add(name)
    try:
        from backend.llm.nl_fallback import parse

        for name, _ in parse(session, message):
            add(name)
    except Exception:  # the parser is a hint only
        pass
    # two or more column names of this session → the request is about a column correspondence
    try:
        from backend.llm.nl_fallback import _mentions_columns

        if len(_mentions_columns(session, message)) >= 2:
            add("explain_match")
            add("decide_mapping")
    except Exception:
        pass
    msg = _tokens(message)
    scored = sorted(((len(msg & toks), name) for name, toks in _TOOL_TOKENS.items() if name not in chosen), key=lambda x: (-x[0], x[1]))
    for score, name in scored:
        if len(chosen) >= max_tools or score == 0:
            break
        add(name)
    return chosen


_ACTION_WORDS = {
    "execute_merge": {"merge", "combine", "join", "integrate", "unify"},
    "generate_merge_plan": {"plan", "steps", "walk", "approach", "merge", "combine"},
    "set_preferences": {"only", "above", "below", "percent", "threshold", "prefer", "primary", "identifier", "uncertain", "conservative", "strict", "permissive", "trust", "keep", "priority", "mode", "strategy"},
    "undo_merge": {"undo", "revert", "rollback", "scrap"},
    "decide_mapping": {"approve", "reject", "wrong", "incorrect"},
    "label_entity_link": {"label", "same", "different"},
}


def requested_mutations(session: Any, message: str) -> set[str]:
    """Data-changing tools this message asks for (parser intents ∪ action words); used to gate follow-up rounds."""
    out: set[str] = set()
    try:
        from backend.llm.nl_fallback import parse

        out |= {name for name, _ in parse(session, message) if TOOLS.get(name) and TOOLS[name].mutating}
    except Exception:
        pass
    toks = _tokens(message)
    out |= {name for name, words in _ACTION_WORDS.items() if toks & words}
    return out

