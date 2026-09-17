"""Merge modes (strict / balanced / permissive) and user overrides → effective thresholds."""
from __future__ import annotations

from typing import Any

from backend.core.errors import ToolArgumentError

MODES = ("strict", "balanced", "permissive")


def effective_thresholds(config: dict, mode: str, overrides: dict[str, Any] | None = None) -> dict[str, float]:
    if mode not in MODES:
        raise ToolArgumentError(f"Unknown merge mode {mode!r}; expected one of {MODES}")
    t = dict(config["merge_modes"][mode])
    # older configs without record-link thresholds fall back to the schema-level ones
    t.setdefault("entity_merge", t["auto_merge"])
    t.setdefault("entity_review", t["review"])
    overrides = overrides or {}
    if overrides.get("confidence_threshold") is not None:
        c = float(overrides["confidence_threshold"])
        if not 0 < c <= 1:
            raise ToolArgumentError("confidence_threshold must be in (0, 1]")
        # "only merge matches above X": applies to relationships and record links alike
        t["auto_merge"] = c
        t["min_edge"] = max(t["min_edge"], c)
        t["review"] = min(t["review"], c)
        # record-link probabilities are calibrated, so "above X" is applied to them literally
        t["entity_merge"] = c
        t["entity_review"] = min(t["entity_review"], c)
    if overrides.get("merge_uncertain") is False:
        # "don't merge uncertain records": anything below HIGH is kept separate
        t["auto_merge"] = max(t["auto_merge"], config["thresholds"]["high"])
        t["entity_merge"] = max(t["entity_merge"], config["thresholds"]["high"])
    return {k: round(v, 4) for k, v in t.items()}
