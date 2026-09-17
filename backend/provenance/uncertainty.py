"""Uncertainty propagation from record linkage into aggregates (probabilistic provenance).

Two kinds of uncertainty reach an output row, and they behave differently:

* **record-level** ``_match_probability`` p_i: the probability that the records
  fused into this row really are one entity (Fellegi–Sunter link probability of
  its cluster; 1 for rows with no probabilistic link). Errors are roughly
  independent across rows.
* **schema-level** ``_relationship_confidence``: the product of the confidences
  of the relationships (lookups) the row went through. If a relationship is wrong,
  *every* row using it is wrong together, so this is a correlated scenario
  probability, not a per-row one. It is reported, not folded into the variance.

Treat "row i is correctly linked" as Z_i ~ Bernoulli(p_i), independent. For an
additive aggregate of a measure x over a group G:

    point estimate     S      = Σ_{i∈G} x_i                (what a plain GROUP BY reports)
    expected value     E[S*]  = Σ p_i x_i                  (count: Σ p_i)
    variance           Var    = Σ p_i (1 − p_i) x_i²
    95% interval       E[S*] ± 1.96 √Var                   (normal approximation)
    certain-only       Σ_{p_i ≥ τ} x_i                     (rows at or above the auto-merge threshold)

where S* = Σ Z_i x_i counts only correctly linked rows. For ``avg`` the expected
value is the ratio Σ p x / Σ p (no interval is reported). The gap between the
point estimate and E[S*] is how much of a figure rests on uncertain links.
"""
from __future__ import annotations

from typing import Any

import duckdb

from backend.core.errors import ToolArgumentError
from backend.storage.duck import quote_ident, quote_literal

AGGS = ("sum", "count", "avg")


def aggregate_with_uncertainty(parquet_path: str, measure: str | None = None, group_by: str | None = None, agg: str = "sum",
                               certain_threshold: float = 0.9, limit: int = 50, scenarios: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if agg not in AGGS:
        raise ToolArgumentError(f"agg must be one of {AGGS}")
    src = f"read_parquet({quote_literal(parquet_path)})"
    con = duckdb.connect(":memory:")
    try:
        cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        if "_match_probability" not in cols:
            raise ToolArgumentError("This merge output has no _match_probability column; re-run the merge")
        for name, c in (("measure", measure), ("group_by", group_by)):
            if c is not None and c not in cols:
                raise ToolArgumentError(f"Unknown {name} column {c!r}")
        if agg != "count" and measure is None:
            raise ToolArgumentError(f"agg={agg} needs a measure column")
        x = "CAST(1 AS DOUBLE)" if agg == "count" else f"TRY_CAST({quote_ident(measure)} AS DOUBLE)"  # type: ignore[arg-type]
        p = "_match_probability"
        g = quote_ident(group_by) if group_by else "'all rows'"
        tau = float(certain_threshold)
        rel = "_relationship_confidence" if "_relationship_confidence" in cols else "CAST(1 AS DOUBLE)"
        rows_sql = f"SELECT {g} AS grp, {x} AS x, {p} AS p, {rel} AS rc FROM {src}"
        rc = "min(rc)"
        if agg == "avg":
            select = f"""grp, count(x) AS n, avg(x) AS point, sum(p * x) / NULLIF(sum(p) FILTER (WHERE x IS NOT NULL), 0) AS expected,
                         NULL AS variance, avg(x) FILTER (WHERE p >= {tau}) AS certain_only, avg(p) AS mean_probability, {rc} AS relationship_confidence"""
        else:
            select = f"""grp, count(x) AS n, sum(x) AS point, sum(p * x) AS expected, sum(p * (1 - p) * x * x) AS variance,
                         coalesce(sum(x) FILTER (WHERE p >= {tau}), 0) AS certain_only, avg(p) AS mean_probability, {rc} AS relationship_confidence"""
        cur = con.execute(f"SELECT {select} FROM ({rows_sql}) WHERE x IS NOT NULL GROUP BY grp ORDER BY point DESC NULLS LAST LIMIT {int(limit)}")
        names = [d[0] for d in cur.description]
        groups = []
        for r in cur.fetchall():
            d = dict(zip(names, r))
            var = d.pop("variance")
            sd = var ** 0.5 if var is not None else None
            groups.append({
                "group": d["grp"] if not isinstance(d["grp"], (bytes, bytearray)) else str(d["grp"]),
                "rows": d["n"], "point": _r(d["point"]), "expected": _r(d["expected"]),
                "ci95_low": _r(d["expected"] - 1.96 * sd) if sd is not None and d["expected"] is not None else None,
                "ci95_high": _r(d["expected"] + 1.96 * sd) if sd is not None and d["expected"] is not None else None,
                "certain_only": _r(d["certain_only"]), "mean_probability": _r(d["mean_probability"], 4),
                "min_relationship_confidence": _r(d["relationship_confidence"], 4),
            })
        # schema-level scenarios: "if relationship R is wrong", rows that were joined through R drop out of the figure.
        # Relationship errors are shared by all their rows, so this is a what-if with probability 1 - confidence,
        # not something to fold into the per-row variance.
        scenario_out = []
        for sc in scenarios or []:
            if sc.get("flag_column") in cols:
                depends = f"coalesce({quote_ident(sc['flag_column'])}, false)"
            elif sc.get("src_row_column") in cols:
                depends = f"{quote_ident(sc['src_row_column'])} IS NOT NULL"
            else:
                continue
            fn = {"sum": "sum(xx)", "count": "count(xx)", "avg": "avg(xx)"}[agg]
            got = dict(con.execute(
                f"SELECT {g} AS grp, {fn} FROM (SELECT *, {x} AS xx FROM {src}) "
                f"WHERE xx IS NOT NULL AND NOT ({depends}) GROUP BY grp").fetchall())
            scenario_out.append({"relationship": sc["relationship"], "probability_wrong": round(1 - float(sc["confidence"]), 4),
                                 "if_wrong": {str(gr["group"]): _r(got.get(gr["group"], 0.0 if agg != "avg" else None)) for gr in groups}})
        total = con.execute(f"SELECT count(*), sum({p}), count(*) FILTER (WHERE {p} < {tau}) FROM {src}").fetchone()
    finally:
        con.close()
    return {
        "agg": agg, "measure": measure, "group_by": group_by, "certain_threshold": tau, "groups": groups,
        "rows": total[0], "expected_correct_rows": _r(total[1]), "rows_below_threshold": total[2], "scenarios": scenario_out,
        "method": "rows ~ independent Bernoulli(_match_probability); expected = Σp·x, variance = Σp(1−p)x², 95% normal interval. "
                  "Relationship (schema-level) confidence is correlated across rows and reported separately as min_relationship_confidence.",
    }


def _r(v: Any, digits: int = 3) -> Any:
    return round(float(v), digits) if v is not None else None
