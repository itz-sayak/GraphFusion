"""Value transformations executed inside DuckDB (vectorised, streaming).

Transformations are *planned* from profiles (so they appear in the merge plan
and in provenance before anything runs) and *compiled* to SQL expressions:

* ``parse_date``      mixed-format date strings → DATE/TIMESTAMP via ``try_strptime`` with an
                      ordered format list (day-first formats first when the column is day-first);
* ``parse_currency``  '₹1,200.50' / 'USD 45' / '45 EUR' → amount, ISO currency and base-currency
                      amount using the dated reference rates in ``config/fx_rates.yaml``.

The raw source value is always preserved in ``<column>_raw``.
"""
from __future__ import annotations

from typing import Any

from backend.config import fx_rates
from backend.core.models import DataType, DatasetProfile, SemanticType
from backend.core.values import iso_currency_codes
from backend.storage.duck import quote_ident, quote_literal

_DAYFIRST = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"]
_MONTHFIRST = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]
_TEXTUAL = ["%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y%m%d"]
_DATETIME = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]


def plan_transformations(profile: DatasetProfile, base_currency: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    currency_code_col = next((c.name for c in profile.columns if c.semantic_type == SemanticType.CURRENCY_CODE), None)
    for col in profile.columns:
        if col.data_type == DataType.STRING and col.semantic_type in (SemanticType.DATE, SemanticType.DATETIME):
            dayfirst = bool(col.date_dayfirst) if col.date_dayfirst is not None else col.date_format in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y %H:%M")
            out.append(
                {
                    "column": col.name,
                    "transform": "parse_date",
                    "dayfirst": dayfirst,
                    "has_time": col.semantic_type == SemanticType.DATETIME,
                    "dominant_format": col.date_format,
                    "description": f"parse mixed-format dates ({'day-first' if dayfirst else 'month-first'} disambiguation) into ISO {'timestamps' if col.semantic_type == SemanticType.DATETIME else 'dates'}",
                }
            )
        elif col.semantic_type == SemanticType.CURRENCY and col.data_type == DataType.STRING:
            out.append(
                {
                    "column": col.name,
                    "transform": "parse_currency",
                    "base_currency": base_currency,
                    "currency_column": currency_code_col,
                    "outputs": [f"{col.name}_amount", f"{col.name}_currency", f"{col.name}_{base_currency.lower()}"],
                    "description": f"split mixed-currency text into amount + ISO currency and convert to {base_currency} (reference rates as of {fx_rates()['as_of']})",
                }
            )
        elif col.semantic_type == SemanticType.CURRENCY and col.data_type in (DataType.FLOAT, DataType.INTEGER) and currency_code_col:
            out.append(
                {
                    "column": col.name,
                    "transform": "convert_currency",
                    "base_currency": base_currency,
                    "currency_column": currency_code_col,
                    "outputs": [f"{col.name}_{base_currency.lower()}"],
                    "description": f"convert amounts to {base_currency} using the row's {currency_code_col}",
                }
            )
    return out


def _fmt_list(formats: list[str]) -> str:
    return "[" + ", ".join(quote_literal(f) for f in formats) + "]"


def date_sql(col: str, dayfirst: bool, has_time: bool) -> str:
    q = f"trim(CAST({quote_ident(col)} AS VARCHAR))"
    dates = (_DAYFIRST if dayfirst else _MONTHFIRST) + _TEXTUAL
    ts = f"coalesce(try_strptime({q}, {_fmt_list(_DATETIME)}), try_strptime({q}, {_fmt_list(dates)}))"
    return ts if has_time else f"CAST({ts} AS DATE)"


def _currency_code_sql(text: str, currency_column: str | None) -> str:
    codes = sorted(iso_currency_codes())
    iso = f"nullif(regexp_extract(upper({text}), '\\b({'|'.join(codes)})\\b', 1), '')"
    symbol_cases = " ".join(
        f"WHEN contains({text}, {quote_literal(sym)}) THEN {quote_literal(code)}"
        for sym, code in sorted(fx_rates()["symbols"].items(), key=lambda kv: -len(kv[0]))
    )
    fallback = f"upper(CAST({quote_ident(currency_column)} AS VARCHAR))" if currency_column else "NULL"
    return f"coalesce({iso}, CASE {symbol_cases} ELSE NULL END, {fallback})"


def _rate_case(code_sql: str, base: str) -> str:
    rates = fx_rates()["to_usd"]
    base_rate = rates.get(base, 1.0)
    cases = " ".join(f"WHEN {quote_literal(c)} THEN {r / base_rate!r}" for c, r in rates.items())
    return f"CASE {code_sql} {cases} ELSE NULL END"


def currency_sql(col: str, base: str, currency_column: str | None) -> dict[str, str]:
    text = f"CAST({quote_ident(col)} AS VARCHAR)"
    amount = f"TRY_CAST(replace(regexp_extract({text}, '-?[0-9][0-9,]*(\\.[0-9]+)?', 0), ',', '') AS DOUBLE)"
    code = _currency_code_sql(text, currency_column)
    return {"amount": amount, "currency": code, "converted": f"round({amount} * ({_rate_case(code, base)}), 4)"}


def convert_sql(col: str, base: str, currency_column: str) -> str:
    code = f"upper(CAST({quote_ident(currency_column)} AS VARCHAR))"
    return f"round(CAST({quote_ident(col)} AS DOUBLE) * ({_rate_case(code, base)}), 4)"


def compile_select(columns: list[str], transforms: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """SELECT list for a transformed dataset view and a map output column → transformation record."""
    by_col = {t["column"]: t for t in transforms}
    select: list[str] = []
    lineage: dict[str, dict[str, Any]] = {}
    for c in columns:
        t = by_col.get(c)
        q = quote_ident(c)
        if t is None:
            select.append(q)
            continue
        if t["transform"] == "parse_date":
            select.append(f"{date_sql(c, t['dayfirst'], t['has_time'])} AS {q}")
            select.append(f"{q} AS {quote_ident(c + '_raw')}")
            lineage[c] = {"transform": "parse_date", "source_column": c}
            lineage[c + "_raw"] = {"transform": None, "source_column": c}
        elif t["transform"] == "parse_currency":
            parts = currency_sql(c, t["base_currency"], t.get("currency_column"))
            a, cur, conv = t["outputs"]
            select += [f"{parts['amount']} AS {quote_ident(a)}", f"{parts['currency']} AS {quote_ident(cur)}", f"{parts['converted']} AS {quote_ident(conv)}", f"{q} AS {quote_ident(c + '_raw')}"]
            for o, kind in ((a, "parse_currency.amount"), (cur, "parse_currency.currency"), (conv, f"parse_currency.convert_to_{t['base_currency']}")):
                lineage[o] = {"transform": kind, "source_column": c}
            lineage[c + "_raw"] = {"transform": None, "source_column": c}
        elif t["transform"] == "convert_currency":
            select.append(q)
            (conv,) = t["outputs"]
            select.append(f"{convert_sql(c, t['base_currency'], t['currency_column'])} AS {quote_ident(conv)}")
            lineage[conv] = {"transform": f"convert_currency_to_{t['base_currency']}", "source_column": c}
    return select, lineage
