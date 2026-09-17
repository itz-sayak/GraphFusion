"""LLM-proposed value crosswalks, verified on the data.

Some correspondences cannot be found from names, formats or overlapping values
because the two columns encode the same real-world things with *different
vocabularies*: "Manhattan" vs "New York County, New York", "CA" vs "California",
"DEU" vs "Germany". A static value map (``config/value_maps.yaml``) only covers
what someone wrote down. Here the language model proposes the mapping, but it is
never trusted:

    1. candidates   pairs of text columns from different datasets whose value sets are fully
                    known (complete sketches), small enough to show, and not already equal
                    under an existing normaliser (value overlap < 0.5)
    2. proposal     the model sees both distinct-value lists and returns {value_A: value_B | null}
    3. verification every proposed value must exist in its column (hallucinations reject the
                    crosswalk above 20%); at least 2 non-trivial pairs; coverage of the smaller
                    column >= 0.5; near one-to-one (distinct targets / mapped >= 0.8)
    4. use          a verified crosswalk becomes a runtime value normaliser "crosswalk:<id>" that is
                    offered to the scorer for that column pair only; the normal 7-signal score,
                    alignment and relationship inference still decide whether the columns match

Proposals are cached by the value lists, so repeated discoveries spend no tokens.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.core.models import DataType, DatasetProfile, SemanticType
from backend.core.values import basic_normalize, register_normalizer
from backend.logging_conf import get_logger
from backend.profiling.sketches import ColumnSketch, overlap_stats

log = get_logger(__name__)

_NOT_VOCABULARY = {SemanticType.ID, SemanticType.EMAIL, SemanticType.PHONE, SemanticType.DATE, SemanticType.DATETIME, SemanticType.ZIPCODE,
                   SemanticType.CURRENCY, SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.LATITUDE, SemanticType.LONGITUDE}

Proposer = Callable[[dict[str, Any]], dict[str, str | None]]

PROMPT = """Two columns from different datasets may name the same real-world things with different vocabularies
(abbreviations vs full names, codes vs names, local vs official names).

Column A: {left_name} (dataset {left_ds}) values: {left_values}
Column B: {right_name} (dataset {right_ds}) values: {right_values}

For each value in A, give the value in B that denotes the SAME real-world thing, or null if none does.
Use only values exactly as listed. Do not guess: if the columns describe different kinds of things, map everything to null.
Answer with one JSON object {{"<value from A>": "<value from B>" or null, ...}} and nothing else."""


@dataclass
class Crosswalk:
    name: str
    left: str  # column key "dataset::column" (the smaller vocabulary)
    right: str
    mapping: dict[str, str]  # basic-normalised A value -> basic-normalised B value (verified pairs only)
    coverage: float
    injectivity: float
    proposed: int
    hallucinated: int
    accepted: bool
    reason: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalizer_for(mapping: dict[str, str]) -> Callable[[Any], str | None]:
    def _norm(value: Any) -> str | None:
        b = basic_normalize(value)
        return mapping.get(b, b) if b is not None else None

    return _norm


def register(cw: Crosswalk) -> None:
    register_normalizer(cw.name, normalizer_for(cw.mapping))


def candidate_pairs(profiles: dict[str, DatasetProfile], sketches: dict[str, dict[str, ColumnSketch]], cfg: dict,
                    accepted: set[frozenset[str]], scores: dict[frozenset[str], float]) -> list[tuple[str, str]]:
    small_max, large_max = int(cfg.get("max_values_small", 80)), int(cfg.get("max_values_large", 400))
    cols = []
    for ds, prof in profiles.items():
        for c in prof.columns:
            sk = sketches.get(ds, {}).get(c.name)
            if sk is None or not sk.complete or c.data_type != DataType.STRING or c.semantic_type in _NOT_VOCABULARY:
                continue
            if 3 <= len(sk.values) <= large_max:
                cols.append((ds, c.name, sk))
    out = []
    for i, (da, ca, sa) in enumerate(cols):
        for db, cb, sb in cols[i + 1:]:
            if da == db:
                continue
            (la, lsk), (rb, rsk) = sorted([((f"{da}::{ca}"), sa), ((f"{db}::{cb}"), sb)], key=lambda x: (len(x[1].values), x[0]))
            if len(lsk.values) > small_max:
                continue
            pair = frozenset((la, rb))
            if pair in accepted:
                continue
            ov = overlap_stats(lsk, rsk)
            if max(ov["containment_left"], ov["containment_right"]) >= 0.5:
                continue  # already comparable without a crosswalk
            out.append((la, rb))
    out.sort(key=lambda p: (-scores.get(frozenset(p), 0.0), p))
    return out[: int(cfg.get("max_pairs", 6))]


def verify(left: str, right: str, left_values: list[str], right_values: list[str], proposal: dict[str, Any], cfg: dict) -> Crosswalk:
    lset = {basic_normalize(v): v for v in left_values}
    rset = {basic_normalize(v) for v in right_values}
    mapping: dict[str, str] = {}
    proposed = hallucinated = 0
    for a, b in (proposal or {}).items():
        if b is None or str(b).strip().lower() in ("", "null", "none"):
            continue
        proposed += 1
        na, nb = basic_normalize(a), basic_normalize(b)
        if na not in lset or nb not in rset:
            hallucinated += 1
            continue
        mapping[na] = nb
    nontrivial = {a: b for a, b in mapping.items() if a != b}
    coverage = len(mapping) / max(1, len(lset))
    injectivity = len(set(mapping.values())) / max(1, len(mapping))
    name = "crosswalk:" + hashlib.sha1(f"{left}|{right}|{sorted(mapping.items())}".encode()).hexdigest()[:10]
    reason = "verified"
    if proposed and hallucinated / proposed > float(cfg.get("max_hallucinated", 0.2)):
        reason = f"{hallucinated} of {proposed} proposed values do not exist in the data"
    elif len(nontrivial) < 2:
        reason = "no mapping between different values (nothing a crosswalk would add)"
    elif coverage < float(cfg.get("min_coverage", 0.5)):
        reason = f"covers only {coverage:.0%} of the smaller vocabulary"
    elif injectivity < float(cfg.get("min_injectivity", 0.8)):
        reason = f"not one-to-one ({injectivity:.0%} distinct targets)"
    return Crosswalk(name, left, right, mapping, round(coverage, 4), round(injectivity, 4), proposed, hallucinated, reason == "verified", reason)


def llm_proposer(provider: Any) -> Proposer:
    def _propose(req: dict[str, Any]) -> dict[str, str | None]:
        msg = PROMPT.format(**{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v) for k, v in req.items()})
        resp = provider.chat([{"role": "user", "content": msg}], tools=None, temperature=0.0, max_tokens=2048)
        text = resp.content or ""
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    return _propose


def propose_crosswalks(profiles: dict[str, DatasetProfile], sketches: dict[str, dict[str, ColumnSketch]], config: dict,
                       accepted: set[frozenset[str]], scores: dict[frozenset[str], float], proposer: Proposer | None,
                       cache_path: Path | None = None) -> list[Crosswalk]:
    cfg = config["schema_matching"].get("crosswalk", {})
    if not cfg.get("enabled", False) or proposer is None:
        return []
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    out: list[Crosswalk] = []
    for left, right in candidate_pairs(profiles, sketches, cfg, accepted, scores):
        (lds, lcol), (rds, rcol) = left.split("::", 1), right.split("::", 1)
        lvals, rvals = sorted(sketches[lds][lcol].values), sorted(sketches[rds][rcol].values)
        req = {"left_name": lcol, "left_ds": lds, "left_values": lvals, "right_name": rcol, "right_ds": rds, "right_values": rvals}
        key = hashlib.sha1(json.dumps(req, sort_keys=True).encode()).hexdigest()
        if key in cache:
            proposal = cache[key]
        else:
            try:
                proposal = proposer(req)
            except Exception as exc:  # noqa: BLE001 - an unavailable model must not break discovery
                log.warning("Crosswalk proposal failed", left=left, right=right, error=repr(exc))
                continue
            cache[key] = proposal
        cw = verify(left, right, lvals, rvals, proposal, cfg)
        log.info("Crosswalk checked", left=left, right=right, accepted=cw.accepted, reason=cw.reason, pairs=len(cw.mapping))
        if cw.accepted:
            register(cw)
        out.append(cw)
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    return out
