"""Value-level normalisation shared by profiling, matching and merging.

Keeping these in one place guarantees that the normaliser that *measured*
value overlap during matching is exactly the transformation *applied* during
the merge, and is recorded under the same name in provenance.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Callable, Iterable

from backend.config import fx_rates, value_maps

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_TRAILING_ZERO = re.compile(r"^(-?\d+)\.0+$")


def basic_normalize(value: Any) -> str | None:
    if isinstance(value, str):
        return _basic_str(value)
    return _basic_any(value)


@lru_cache(maxsize=500_000)
def _basic_str(value: str) -> str | None:
    return _basic_any(value)


def _basic_any(value: Any) -> str | None:
    """Case/whitespace/accent-insensitive canonical string; integral floats collapse (5.0 → 5)."""
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if value.is_integer():
            return str(int(value))
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _WS.sub(" ", s.strip().lower())
    m = _TRAILING_ZERO.match(s)
    if m:
        s = m.group(1)
    return s or None


def alnum_normalize(value: Any) -> str | None:
    """Drop punctuation as well: 'Claremont/Bathgate' ≡ 'claremont bathgate'."""
    s = basic_normalize(value)
    if s is None:
        return None
    s = _WS.sub(" ", _PUNCT.sub(" ", s)).strip()
    return s or None


def id_normalize(value: Any) -> str | None:
    """Identifier canonical form: strip leading zeros and alpha prefixes like 'C-00123' → '123'."""
    s = basic_normalize(value)
    if s is None:
        return None
    digits = re.sub(r"^[a-z]+[-_ ]?", "", s)
    if digits.isdigit():
        return str(int(digits))
    return s


def digits_normalize(value: Any, keep_last: int = 10) -> str | None:
    """Phone-style canonical form: digits only, last ``keep_last`` digits (drops country codes)."""
    if value is None:
        return None
    d = re.sub(r"\D", "", str(value))
    if len(d) < 6:
        return None
    return d[-keep_last:]


@lru_cache
def _geo_suffix_patterns() -> list[re.Pattern[str]]:
    return [re.compile(p) for p in value_maps().get("geo_suffixes", [])]


def geo_normalize(value: Any) -> str | None:
    s = alnum_normalize(value) if value is not None else None
    if s is None:
        return None
    raw = basic_normalize(value) or s
    for pat in _geo_suffix_patterns():
        raw = pat.sub("", raw).strip()
    return alnum_normalize(raw)


def make_domain_normalizer(domain: str) -> Callable[[Any], str | None]:
    mapping = {alnum_normalize(k): alnum_normalize(v) for k, v in value_maps()["domains"][domain].items()}

    state_suffix = _geo_suffix_patterns()[0] if _geo_suffix_patterns() else None

    @lru_cache(maxsize=200_000)
    def _cached(value: str) -> str | None:
        return _norm_impl(value)

    def _norm(value: Any) -> str | None:
        return _cached(value) if isinstance(value, str) else _norm_impl(value)

    def _norm_impl(value: Any) -> str | None:
        s = geo_normalize(value)
        if s is None:
            return None
        # most specific form first: 'new york county, ny' → 'new york county' → 'new york'
        full = basic_normalize(value) or ""
        if state_suffix is not None:
            full = state_suffix.sub("", full).strip()
        for candidate in (alnum_normalize(value), alnum_normalize(full), s):
            if candidate in mapping:
                return mapping[candidate]
        return s

    _norm.__name__ = f"domain:{domain}"
    return _norm


@lru_cache
def normalizers() -> dict[str, Callable[[Any], str | None]]:
    """All registered value normalisers, keyed by the name recorded in evidence/provenance."""
    out: dict[str, Callable[[Any], str | None]] = {
        "basic": basic_normalize,
        "alnum": alnum_normalize,
        "identifier": id_normalize,
        "digits": digits_normalize,
        "geo": geo_normalize,
    }
    for domain in value_maps().get("domains", {}):
        out[f"domain:{domain}"] = make_domain_normalizer(domain)
    by_type: dict[str, list[str]] = {}
    for domain, stype in value_maps().get("domain_semantic_types", {}).items():
        by_type.setdefault(stype, []).append(domain)
    for stype, domains in by_type.items():
        if len(domains) > 1:
            out[f"semantic:{stype}"] = make_composite_normalizer(domains)
    return out


def make_composite_normalizer(domains: list[str]) -> Callable[[Any], str | None]:
    """Apply several reference domains of the same semantic type (first mapping wins)."""
    parts = [make_domain_normalizer(d) for d in domains]

    def _norm(value: Any) -> str | None:
        base = geo_normalize(value)
        for fn in parts:
            mapped = fn(value)
            if mapped is not None and mapped != base:
                return mapped
        return base

    return _norm


_RUNTIME_NORMALIZERS: dict[str, Callable[[Any], str | None]] = {}


def register_normalizer(name: str, fn: Callable[[Any], str | None]) -> None:
    """Register a normaliser created at runtime (e.g. a verified value crosswalk)."""
    _RUNTIME_NORMALIZERS[name] = fn


def get_normalizer(name: str) -> Callable[[Any], str | None]:
    if name in ("identity", "basic"):
        return basic_normalize
    if name in _RUNTIME_NORMALIZERS:
        return _RUNTIME_NORMALIZERS[name]
    return normalizers()[name]


# --------------------------------------------------------------------------- dates

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y%m%d",
]
DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M:%S %p",
]


def _try_formats(s: str, formats: Iterable[str]) -> tuple[datetime, str] | None:
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt), fmt
        except ValueError:
            continue
    return None


def detect_date_formats(values: Iterable[Any]) -> tuple[float, str | None, bool]:
    """Return (parse_rate, dominant_format, has_time) for a sample of values.
    Use :func:`is_dayfirst` for the day/month order decision.

    Day/month ambiguity is resolved at column level: if any value only parses
    day-first (e.g. 23/04/2024) the column is treated as day-first, which is
    how a human analyst would disambiguate ``04/05/2024`` in the same column.
    """
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals:
        return 0.0, None, False
    counts: Counter[str] = Counter()
    parsed = 0
    has_time = False
    for s in vals:
        hit = _try_formats(s, DATETIME_FORMATS)
        if hit:
            counts[hit[1]] += 1
            parsed += 1
            has_time = True
            continue
        hit = _try_formats(s, DATE_FORMATS)
        if hit:
            counts[hit[1]] += 1
            parsed += 1
    dayfirst = any(_try_formats(s, ["%d/%m/%Y", "%d-%m-%Y"]) and not _try_formats(s, ["%m/%d/%Y", "%m-%d-%Y"]) for s in vals)
    if dayfirst:
        for us_fmt, eu_fmt in (("%m/%d/%Y", "%d/%m/%Y"), ("%m-%d-%Y", "%d-%m-%Y")):
            counts[eu_fmt] += counts.pop(us_fmt, 0)
    fmt = counts.most_common(1)[0][0] if counts else None
    return parsed / len(vals), fmt, has_time


def is_dayfirst(values: Iterable[Any]) -> bool:
    """True when at least one slash/dash date only parses day-first (e.g. 23/04/2024)."""
    for v in values:
        s = str(v).strip() if v is not None else ""
        if s and _try_formats(s, ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]) and not _try_formats(s, ["%m/%d/%Y", "%m-%d-%Y"]):
            return True
    return False


def parse_date(value: Any, dayfirst: bool = False) -> datetime | None:
    """Parse one value with the multi-format parser; ``dayfirst`` decides dd/mm vs mm/dd."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    s = str(value).strip()
    if not s:
        return None
    hit = _try_formats(s, DATETIME_FORMATS)
    if hit:
        return hit[0].replace(tzinfo=None)
    order = list(DATE_FORMATS)
    if dayfirst:
        order.remove("%d/%m/%Y")
        order.remove("%d-%m-%Y")
        order[1:1] = ["%d/%m/%Y", "%d-%m-%Y"]
    hit = _try_formats(s, order)
    return hit[0] if hit else None


# --------------------------------------------------------------------------- currency

_AMOUNT = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ISO_CODE = re.compile(r"\b([A-Z]{3})\b")


def iso_currency_codes() -> set[str]:
    return set(fx_rates()["to_usd"].keys()) | {"CAD", "AUD", "CHF", "CNY", "SGD", "HKD", "SEK", "NOK", "MXN", "BRL", "ZAR"}


def parse_currency(value: Any, default_code: str | None = None) -> tuple[float | None, str | None]:
    """Parse '₹1,200.50', 'USD 45', '45.00 EUR', 12.5 → (amount, ISO code)."""
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return (None, None) if value != value else (float(value), default_code)
    s = str(value).strip()
    if not s:
        return None, None
    code = None
    m = _ISO_CODE.search(s.upper())
    if m and m.group(1) in iso_currency_codes():
        code = m.group(1)
    if code is None:
        for sym, iso in sorted(fx_rates()["symbols"].items(), key=lambda kv: -len(kv[0])):
            if sym in s:
                code = iso
                break
    num = _AMOUNT.search(s.replace(" ", ""))
    if not num:
        return None, code or default_code
    return float(num.group(0).replace(",", "")), code or default_code


def to_base_currency(amount: float | None, code: str | None, base: str = "USD") -> float | None:
    if amount is None:
        return None
    rates = fx_rates()["to_usd"]
    if code is None or code == base:
        return amount
    if code not in rates or base not in rates:
        return None
    return round(amount * rates[code] / rates[base], 4)


# --------------------------------------------------------------------------- shapes


def value_shape(value: Any, max_len: int = 12) -> str:
    """Collapse character classes into a compact shape: 'C-8921' → 'A-9', 'rahul@x.com' → 'a@a.a'."""
    s = str(value)[:64]
    out: list[str] = []
    for ch in s:
        if ch.isdigit():
            c = "9"
        elif ch.isalpha():
            c = "A" if ch.isupper() else "a"
        elif ch.isspace():
            c = " "
        else:
            c = ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)[:max_len]
