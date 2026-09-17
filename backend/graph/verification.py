"""Row-level verification of attribute correspondences between key-joined datasets.

Schema matching scores a column pair from profiles and sketches: names, types,
formats and value *distributions*. Two columns can look alike on all of those
and still hold different facts: two date columns from the same period, two
count columns, or two random-looking identifiers. When the datasets are joined
by a verified key, there is a direct test: if A.x and B.y carry the same
information, then for a row of A and its joined row of B, ``x`` and ``y`` agree.

For every lookup / entity-key-merge relationship:

    sample up to N rows of the referencing side (DuckDB reservoir sample)
    join them to the referenced side on the key (the key match's normaliser)
    agreement(x, y) = |{joined rows: norm(x) = norm(y)}| / |{joined rows: x, y both non-null}|
    agreement < attribute_min_agreement  →  reject the correspondence

Comparison uses the attribute match's own value normaliser; numbers compare with a
relative tolerance and timestamps at the precision both sides share. Pairs with too
few comparable rows (< 30) are left unverified rather than guessed.
Only key-based relationships are checked: probabilistic entity links and aggregated
lookups have no row-to-row correspondence to test.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from backend.core.models import ColumnMatch, DatasetArtifact, DatasetRelationship, JoinKind
from backend.core.values import get_normalizer
from backend.logging_conf import get_logger
from backend.storage.duck import quote_ident, quote_literal

log = get_logger(__name__)
MIN_COMPARED = 30


def _canon(value: Any, norm) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return ("dt", value.date().isoformat()) if (value.hour, value.minute, value.second, value.microsecond) == (0, 0, 0, 0) else ("dt", value.isoformat(timespec="minutes"))
    if isinstance(value, date):
        return ("dt", value.isoformat())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("num", float(value))
    return ("str", norm(value))


def _agree(a: Any, b: Any) -> bool:
    if a[0] == "num" and b[0] == "num":
        return abs(a[1] - b[1]) <= 1e-6 * max(1.0, abs(a[1]), abs(b[1]))
    if a[0] == "dt" and b[0] == "dt":
        short = min(len(a[1]), len(b[1]))  # compare at the precision both sides share
        return a[1][:short] == b[1][:short]
    if a[0] != b[0]:
        # a number stored as text on one side
        try:
            return abs(float(a[1]) - float(b[1])) <= 1e-6 * max(1.0, abs(float(a[1])))
        except (TypeError, ValueError):
            return str(a[1]) == str(b[1])
    return a[1] == b[1]


def _side(m: ColumnMatch, dataset: str) -> str:
    return m.left.column if m.left.dataset_id == dataset else m.right.column


def verify_attribute_matches(relationships: list[DatasetRelationship], artifacts: dict[str, DatasetArtifact], config: dict) -> list[dict[str, Any]]:
    merge_cfg = config["merge"]
    threshold = float(merge_cfg.get("attribute_min_agreement", 0.5))
    sample_n = int(merge_cfg.get("attribute_verify_sample", 20000))
    report: list[dict[str, Any]] = []
    con = duckdb.connect(":memory:")
    try:
        for rel in relationships:
            if rel.join_kind not in (JoinKind.LOOKUP, JoinKind.ENTITY_KEY_MERGE) or not rel.key_matches or not rel.attribute_matches:
                continue
            ev = rel.evidence or {}
            src = ev.get("fact") if ev.get("fact") in (rel.left_dataset, rel.right_dataset) else rel.left_dataset
            dst = rel.right_dataset if src == rel.left_dataset else rel.left_dataset
            if src not in artifacts or dst not in artifacts:
                continue
            key = max(rel.key_matches, key=lambda m: m.score)
            key_norm = get_normalizer(key.evidence.normalizer)
            ksrc, kdst = _side(key, src), _side(key, dst)
            attrs = list(rel.attribute_matches)
            src_cols = [ksrc] + [_side(m, src) for m in attrs]
            dst_cols = [kdst] + [_side(m, dst) for m in attrs]
            psrc = quote_literal(Path(artifacts[src].storage_path).as_posix())
            pdst = quote_literal(Path(artifacts[dst].storage_path).as_posix())
            src_rows = con.execute(
                f"SELECT {', '.join(quote_ident(c) for c in src_cols)} FROM read_parquet({psrc}) "
                f"WHERE {quote_ident(ksrc)} IS NOT NULL USING SAMPLE reservoir({sample_n} ROWS) REPEATABLE (7)"
            ).fetchall()
            wanted = {key_norm(r[0]) for r in src_rows} - {None}
            if not wanted:
                continue
            # referenced side: only rows whose normalised key was sampled (streamed in batches)
            dst_index: dict[Any, tuple] = {}
            cur = con.execute(f"SELECT {', '.join(quote_ident(c) for c in dst_cols)} FROM read_parquet({pdst}) WHERE {quote_ident(kdst)} IS NOT NULL")
            while True:
                batch = cur.fetchmany(100_000)
                if not batch:
                    break
                for r in batch:
                    k = key_norm(r[0])
                    if k in wanted and k not in dst_index:
                        dst_index[k] = r
            for i, m in enumerate(attrs, start=1):
                norm = get_normalizer(m.evidence.normalizer)
                compared = agreed = 0
                for r in src_rows:
                    other = dst_index.get(key_norm(r[0]))
                    if other is None:
                        continue
                    a, b = _canon(r[i], norm), _canon(other[i], norm)
                    if a is None or b is None or a[1] is None or b[1] is None:
                        continue
                    compared += 1
                    agreed += _agree(a, b)
                entry = {"relationship": f"{rel.left_dataset}--{rel.right_dataset}", "key": f"{ksrc} = {kdst}",
                         "left": m.left.key, "right": m.right.key, "compared_rows": compared}
                if compared < MIN_COMPARED:
                    m.evidence.notes.append(f"row check skipped: only {compared} joined rows with both values")
                    report.append({**entry, "agreement": None, "decision": "unverified"})
                    continue
                agreement = agreed / compared
                m.evidence.row_agreement = round(agreement, 4)
                m.evidence.rows_compared = compared
                if agreement < threshold:
                    m.accepted = False
                    m.rejection_reason = (f"values disagree on joined rows: equal in {agreement:.0%} of {compared:,} rows joined on "
                                          f"{src}.{ksrc} = {dst}.{kdst} (needs {threshold:.0%})")
                    rel.attribute_matches = [x for x in rel.attribute_matches if x is not m]
                    decision = "rejected"
                else:
                    m.evidence.notes.append(f"verified on {compared:,} joined rows: {agreement:.0%} agree")
                    decision = "verified"
                report.append({**entry, "agreement": round(agreement, 4), "decision": decision})
                log.info("Attribute correspondence checked on joined rows", left=m.left.key, right=m.right.key, agreement=round(agreement, 3), rows=compared, decision=decision)
    finally:
        con.close()
    return report
