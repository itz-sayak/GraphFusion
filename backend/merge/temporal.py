"""Effective-dated attribute history (temporal validity) for fused entities.

Conflict resolution picks one value per (entity, attribute). When records carry
timestamps, the disagreeing values are often not errors but *history*: a customer
moved from Pune to Bangalore. This module turns the candidates into validity
intervals, a simplified bitemporal "valid time" model:

    candidates c_1..c_n with timestamps t_1 ≤ … ≤ t_n (ties: source priority wins)
    runs      maximal blocks of consecutive candidates with equal normalised value
    interval  run k is valid on [start_k, start_{k+1});  the last run is open-ended (current)

Timestamp choice per source: a change time ("updated", "modified", "last …") is
preferred; otherwise a creation time ("created", "signup", "joined", …) is used and
the interval is labelled ``timestamp_kind = "creation"``: the value was true *at
least* from then, a weaker statement than a change log. Undated candidates are
kept as rows with NULL validity so nothing disappears.

History is emitted only where it says something: descriptive attributes
(not identifiers, timestamps or raw-text shadows) of an entity whose dated
records hold at least two distinct values.

Invariants (checked by validation ``temporal_consistency``): per (entity,
attribute), valid_from < valid_to, intervals do not overlap, and at most one
interval is current.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

CHANGE_TOKENS = {"updated", "modified", "last", "timestamp", "changed"}
CREATION_TOKENS = {"created", "signup", "joined", "registered", "since", "opened", "creation"}


@dataclass
class DatedValue:
    value: Any
    norm: Any
    dataset: str
    column: str
    row: int
    timestamp: datetime | None
    kind: str | None  # "change" | "creation" | None
    priority: int = 99


def build_intervals(entity_id: str, attribute: str, values: list[DatedValue]) -> list[dict[str, Any]]:
    present = [v for v in values if v.norm is not None]
    dated = [v for v in present if v.timestamp is not None]
    out: list[dict[str, Any]] = []
    if not dated:
        return out
    # one representative per timestamp: the highest-priority source (lowest number)
    by_ts: dict[datetime, DatedValue] = {}
    for v in sorted(dated, key=lambda v: (v.timestamp, v.priority)):
        by_ts.setdefault(v.timestamp, v)  # type: ignore[arg-type]
    ordered = [by_ts[t] for t in sorted(by_ts)]
    runs: list[list[DatedValue]] = []
    for v in ordered:
        if runs and runs[-1][-1].norm == v.norm:
            runs[-1].append(v)
        else:
            runs.append([v])
    for i, run in enumerate(runs):
        first = run[0]
        nxt = runs[i + 1][0].timestamp if i + 1 < len(runs) else None
        out.append({
            "entity_id": entity_id, "attribute": attribute, "value": first.value,
            "valid_from": first.timestamp, "valid_to": nxt, "is_current": nxt is None,
            "timestamp_kind": "change" if any(v.kind == "change" for v in run) else "creation",
            "supporting_records": len(run),
            "source_dataset": first.dataset, "source_column": first.column, "source_row": first.row,
        })
    for v in present:
        if v.timestamp is None:
            out.append({"entity_id": entity_id, "attribute": attribute, "value": v.value, "valid_from": None, "valid_to": None, "is_current": False,
                        "timestamp_kind": "undated", "supporting_records": 1, "source_dataset": v.dataset, "source_column": v.column, "source_row": v.row})
    return out


def check_intervals(history: list[dict[str, Any]]) -> list[str]:
    """Return a list of violations (empty when the history is consistent)."""
    problems: list[str] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for h in history:
        if h["valid_from"] is not None:
            groups.setdefault((h["entity_id"], h["attribute"]), []).append(h)
    for (ent, attr), rows in groups.items():
        rows.sort(key=lambda r: r["valid_from"])
        if sum(1 for r in rows if r["is_current"]) > 1:
            problems.append(f"{ent}.{attr}: more than one current interval")
        for a, b in zip(rows, rows[1:]):
            if a["valid_to"] is None or a["valid_to"] > b["valid_from"]:
                problems.append(f"{ent}.{attr}: intervals overlap at {b['valid_from']}")
        for r in rows:
            if r["valid_to"] is not None and not r["valid_from"] < r["valid_to"]:
                problems.append(f"{ent}.{attr}: empty interval at {r['valid_from']}")
    return problems
