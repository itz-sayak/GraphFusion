"""Conversational agent: natural language → validated tool calls → evidence-based replies.

Loop (for real LLMs): the model receives the system prompt, a live summary of
session state (so users never repeat dataset names), recent conversation and
the tool schemas; it answers with tool calls; the backend validates and runs
them; results are returned to the model; repeat (bounded) until it answers.

Guarantees independent of the model:
* only registered tools run, with schema-validated arguments;
* all figures shown to the user come from deterministic tool renderers;
* if the remote model fails, the turn is transparently re-run with the
  rule-based parser rather than failing the user.
"""
from __future__ import annotations

import json
import time
from typing import Any

from backend.core.errors import LLMProviderError
from backend.llm.base import LLMProvider
from backend.llm.nl_fallback import HELP, parse
from backend.llm.providers.mock import MockProvider
from backend.llm.rate_limit import RateLimitExhausted
from backend.llm.tool_selection import requested_mutations, select_tools
from backend.llm.tools import TOOLS, compact_for_llm, execute_tool
from backend.logging_conf import get_logger
from backend.session.state import Session

log = get_logger(__name__)

SYSTEM_PROMPT = """You are GraphFusion, an expert data-integration assistant.
You help users integrate heterogeneous datasets (CSV, JSON, Parquet, Excel, APIs) by calling tools.

Rules:
1. Every data operation MUST be done by calling a tool. Never invent datasets, columns, numbers, mappings or results.
2. The user already sees each tool's output verbatim. After tools run, reply with at most 3 short sentences: an interpretation or the most useful next step. Do not repeat tables or figures.
3. Use dataset ids and column names exactly as they appear in the session state or tool results.
4. Translate preferences into set_preferences (or the corresponding arguments): "only merge matches above 95%" → confidence_threshold 0.95; "don't merge uncertain records" → merge_uncertain false; "use X as the primary identifier" → primary_key X; "prefer the latest values" → conflict_strategy prefer_latest.
5. "Merge them" / "merge everything" → execute_merge. "Show the plan" → generate_merge_plan. "Find relationships" / "inspect" → find_candidate_relationships.
6. For "why" questions call the matching explain tool (explain_match for columns, explain_relationship for datasets or joins, find_integration_route for routes) and base the answer on its evidence.
7. If a request is ambiguous or unsupported, ask a short clarifying question instead of guessing.
8. Only do what the latest user message asks. Never run a merge, undo, preference change or label because of an earlier message.
9. Earlier results in the conversation may be outdated. Never answer a request for an action or for figures from memory; call the tool.
"""


def state_summary(session: Session) -> str:
    parts = [f"Session {session.session_id}."]
    if session.artifacts:
        parts.append("Datasets: " + "; ".join(f"{d} ({a.source_name}, {a.source_type.value}, {a.row_count} rows, columns: {', '.join(list(a.schema_)[:25])})" for d, a in session.artifacts.items()))
    else:
        parts.append("No datasets loaded.")
    parts.append(f"Relationships discovered: {'no' if session.discovery_stale else 'yes (' + str(len(session.relationships)) + ')'}.")
    parts.append("Preferences: " + json.dumps({k: v for k, v in session.preferences_dict().items() if k != "thresholds"}))
    if session.plan:
        parts.append(f"Current plan: {session.plan.plan_id} rooted at {session.plan.root_dataset}.")
    active = [r for r in session.merges if r.status == "active"]
    outdated = [r for r in session.merges if r.status == "outdated"]
    if outdated and not active:
        parts.append(f"No current merge: the datasets changed after {outdated[-1].result.merge_id}, so it is outdated. A new merge must be executed with execute_merge.")
    if active:
        r = active[-1]
        parts.append(f"Current merge: {r.result.merge_id} with {r.result.row_count} rows, {r.result.conflicts} conflicts; output columns: {', '.join(c for c in r.result.columns if not c.startswith('_'))[:1500]}.")
    return "\n".join(parts)


def suggestions(session: Session) -> list[str]:
    if not session.artifacts:
        return ["Load the sample datasets", "Load the NYC mobility demo"]
    if len(session.artifacts) < 2:
        return ["Profile the dataset", "Show me duplicate entities"]
    if session.discovery_stale:
        return ["Find relationships between the datasets", "Inspect the schemas"]
    if not any(r.status == "active" for r in session.merges):
        return ["Show me the merge plan", "Merge everything", "Only merge matches above 95%", "Show me the graph of dataset relationships"]
    return ["Show me conflicts", "What percentage of rows were matched?", "Generate a data lineage report", "Export the final dataset", "Undo the previous merge"]


class ChatAgent:
    def __init__(self, provider: LLMProvider, max_rounds: int | None = None, history_turns: int | None = None, config: dict | None = None):
        from backend.config import load_config

        cfg = ((config or load_config()).get("llm") or {}).get("agent") or {}
        self.provider = provider
        self.max_rounds = max_rounds or int(cfg.get("max_rounds", 3))
        self.history_turns = history_turns or int(cfg.get("history_turns", 3))
        self.history_chars = int(cfg.get("history_chars", 400))
        self.tool_result_chars = int(cfg.get("tool_result_chars", 1500))
        self.max_tools = int(cfg.get("max_tools", 12))
        self.commentary = str(cfg.get("commentary", "auto"))
        self.last_turn_stats: dict[str, Any] = {}

    def handle(self, session: Session, message: str) -> dict[str, Any]:
        started = time.perf_counter()
        with session.lock:
            session.chat_history.append({"role": "user", "content": message})
            fallback_note = None
            partial: list[dict[str, Any]] = []
            try:
                if self.provider.is_mock:
                    executed, commentary = self._run_rule_based(session, message)
                else:
                    executed, commentary = self._run_llm(session, message, partial)
            except LLMProviderError as exc:
                if partial:
                    # tools already ran: never re-run them (a merge must not execute twice)
                    log.warning("LLM failed after executing tools; returning partial results", error=exc.message)
                    executed, commentary = partial, None
                    fallback_note = f"(The language model stopped responding after {len(partial)} operation(s) — {exc.message[:120]}.)"
                else:
                    log.warning("LLM unavailable; falling back to rule-based parser", error=exc.message)
                    fallback_note = f"(The language model was unavailable — {exc.message[:160]} — so I used the built-in command parser.)"
                    executed, commentary = self._run_rule_based(session, message)

            if executed:
                reply = "\n\n".join(r["rendered"] for r in executed)
                if commentary:
                    reply += "\n\n" + commentary.strip()
            else:
                reply = commentary.strip() if commentary else HELP
            if fallback_note:
                reply += "\n\n" + fallback_note
            cards = [{"type": r["card"], "tool": r["tool"], "data": _card_data(r)} for r in executed if r["ok"] and r.get("card")]
            response = {
                "session_id": session.session_id,
                "reply": reply,
                "tool_calls": [{"tool": r["tool"], "arguments": r.get("arguments", {}), "ok": r["ok"], "error": r.get("error")} for r in executed],
                "cards": cards,
                "suggestions": suggestions(session),
                "provider": self.provider.describe() | ({"fallback": True} if fallback_note else {}),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            session.chat_history.append({"role": "assistant", "content": reply, "tool_calls": response["tool_calls"]})
            session.save_snapshot()
            return response

    # ------------------------------------------------------------------ strategies
    def _run_rule_based(self, session: Session, message: str) -> tuple[list[dict[str, Any]], str | None]:
        calls = parse(session, message)
        executed = []
        for name, args in calls:
            res = execute_tool(session, name, args)
            executed.append(res)
            log.info("Tool executed", tool=name, ok=res["ok"], mode="rule_based")
            if not res["ok"] and TOOLS.get(name) and TOOLS[name].mutating:
                break
        return executed, None if executed else HELP

    def _run_llm(self, session: Session, message: str, executed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
        # token budget per turn (hosted free tiers allow ~8K tokens/minute): short history, relevant tools only,
        # bounded tool results, and follow-up rounds only when they fit the rate limit without waiting
        history = [h for h in session.chat_history[:-1] if h["role"] in ("user", "assistant")][-2 * self.history_turns:]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent session state:\n" + state_summary(session)[:2500]},
            *({"role": h["role"], "content": _history_content(h, self.history_chars)} for h in history),
            {"role": "user", "content": message},
        ]
        names = select_tools(session, message, self.max_tools)
        mutating_ok = requested_mutations(session, message)
        tools = [TOOLS[n].spec() for n in names]
        budget_ok = getattr(self.provider, "budget_ok", None)
        commentary = None
        stats: dict[str, Any] = {"tools_offered": len(tools), "rounds": 0, "skipped_followup": None}
        self.last_turn_stats = stats
        for round_no in range(self.max_rounds):
            if round_no == 0:
                try:
                    resp = self.provider.chat(messages, tools=tools)
                except LLMProviderError as exc:
                    if not _tool_outside_subset(exc) or len(tools) == len(TOOLS):
                        raise
                    # the model asked for a tool that relevance selection left out: offer every tool once
                    stats["retried_with_all_tools"] = True
                    tools = [t.spec() for t in TOOLS.values()]
                    resp = self.provider.chat(messages, tools=tools)
            else:
                failed = any(not r["ok"] for r in executed)
                if self.commentary == "off" and not failed:
                    stats["skipped_followup"] = "commentary disabled"
                    break
                # follow-up rounds may chain further tools ("set the threshold, then merge"), but a data-changing
                # tool is only offered if this message asks for a change; if even that does not fit the rate
                # limit, fall back to commentary without tool schemas, or stop (results are already rendered)
                round_tools = [t for t in tools if not TOOLS[t["name"]].mutating or t["name"] in mutating_ok] or None
                if budget_ok is not None and not budget_ok(messages, round_tools):
                    round_tools = None
                    if not budget_ok(messages, None):
                        stats["skipped_followup"] = "rate-limit budget"
                        log.info("Skipping follow-up LLM round to stay within rate limits", executed=len(executed))
                        break
                try:
                    if budget_ok is not None:
                        resp = self.provider.chat(messages, tools=round_tools, max_tokens=512, max_wait_s=0.0)  # type: ignore[call-arg]
                    else:
                        resp = self.provider.chat(messages, tools=round_tools, max_tokens=512)
                except RateLimitExhausted:
                    stats["skipped_followup"] = "rate-limit budget"
                    break
                except LLMProviderError as exc:
                    if not _tool_outside_subset(exc):
                        raise
                    stats["skipped_followup"] = "model asked for a tool outside this round's subset"
                    break  # the requested work already ran; its rendered results are the reply
            stats["rounds"] += 1
            if not resp.tool_calls and round_no == 0 and mutating_ok & set(names) and not executed:
                # the user asked for an action but the model answered in text (e.g. from stale context): ask once more
                nudge = messages + [{"role": "assistant", "content": resp.content or ""},
                                    {"role": "user", "content": "That request needs a tool call. Call the appropriate tool now instead of answering from earlier results."}]
                if budget_ok is None or budget_ok(nudge, tools):
                    stats["nudged"] = True
                    retry = self.provider.chat(nudge, tools=tools)
                    if retry.tool_calls:
                        messages, resp = nudge, retry
            if not resp.tool_calls:
                commentary = resp.content
                break
            messages.append({
                "role": "assistant", "content": resp.content or "",
                "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in resp.tool_calls],
            })
            for c in resp.tool_calls:
                res = execute_tool(session, c.name, c.arguments)
                executed.append(res)
                log.info("Tool executed", tool=c.name, ok=res["ok"], mode="llm")
                messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": compact_for_llm(res, self.tool_result_chars)})
        return executed, commentary


def _tool_outside_subset(exc: LLMProviderError) -> bool:
    """Providers that validate tool calls server-side reject calls to tools not offered in the request (Groq: HTTP 400)."""
    msg = exc.message.lower()
    return "not in request.tools" in msg or ("tool" in msg and "not in request" in msg)


def _history_content(h: dict[str, Any], limit: int) -> str:
    """Previous turns as the model sees them. Assistant turns are reduced to the tools they ran plus a short
    excerpt, so rendered reports from earlier (possibly outdated) states are not copied into new answers."""
    if h["role"] != "assistant" or not h.get("tool_calls"):
        return h["content"][:limit]
    calls = ", ".join(f"{c['tool']}({'ok' if c.get('ok') else 'failed'})" for c in h["tool_calls"])
    return f"[earlier turn ran: {calls}; its results may be outdated] " + h["content"][: max(0, limit // 3)]


def _card_data(res: dict[str, Any]) -> Any:
    data = res["result"]
    if res["card"] == "profile":
        return data
    if res["card"] == "merge":
        return {k: data[k] for k in ("merge_id", "row_count", "column_count", "columns", "matched_rows", "unmatched_rows", "conflicts", "entity_clusters", "quality_report", "validation", "files", "elapsed_ms")}
    return data


def make_agent(provider: LLMProvider | None = None) -> ChatAgent:
    from backend.llm.factory import make_provider

    try:
        p = provider or make_provider()
    except LLMProviderError as exc:
        log.warning("Configured LLM provider unavailable; using mock", error=exc.message)
        p = MockProvider()
    return ChatAgent(p)

