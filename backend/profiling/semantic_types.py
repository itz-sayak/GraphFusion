"""Semantic type inference.

Each candidate type accumulates evidence from three independent sources, and
the highest-scoring type wins:

* **name prior**   – tokens of the (expanded) column name, e.g. ``email``;
* **value evidence** – fraction of sampled values matching a type-specific
  regex or gazetteer, or statistical signatures (range, uniqueness, length);
* **physical type** – numeric / temporal / string storage type.

Value evidence dominates when strong, so a column called ``col_7`` full of
e-mail addresses is still typed ``EMAIL``, and a column called ``id`` holding
free text is not typed ``ID``.
"""
from __future__ import annotations

import re
from typing import Any

from backend.core.models import DataType, SemanticType
from backend.core.values import detect_date_formats, iso_currency_codes, parse_currency
from backend.matching.normalize import name_tokens

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,20}$")
ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$|^\d{6}$|^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$", re.I)
ALNUM_ID_RE = re.compile(r"^[A-Za-z]{0,4}[-_]?\d{2,}$|^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
# opaque token identifiers: hashes, hex keys, UUIDs without dashes, base62 ids ("e481f51cbdc54678b7cc49136f2d6af7")
TOKEN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{12,64}$")
CURRENCY_TEXT_RE = re.compile(r"^\s*([$₹€£¥]|rs\.?|[A-Z]{3})?\s*-?\d[\d,]*(\.\d+)?\s*([A-Z]{3})?\s*$", re.I)
PERSON_NAME_RE = re.compile(r"^[A-Z][a-zA-Z'.-]+(\s+[A-Z][a-zA-Z'.-]*){1,3}$")

CITY_GAZETTEER = {
    "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad",
    "jaipur", "lucknow", "surat", "kochi", "gurugram", "gurgaon", "noida", "chandigarh", "indore", "bhopal", "nagpur",
    "new york", "new york city", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio",
    "san diego", "dallas", "san jose", "austin", "boston", "seattle", "denver", "miami", "atlanta", "san francisco",
    "london", "paris", "berlin", "madrid", "rome", "tokyo", "singapore", "dubai", "sydney", "toronto", "hong kong",
    "manchester", "amsterdam", "dublin", "zurich", "shanghai", "beijing", "seoul", "bangkok", "jakarta", "manila",
}
COUNTRY_GAZETTEER = {
    "india", "united states", "usa", "us", "united kingdom", "uk", "germany", "france", "japan", "china", "canada",
    "australia", "singapore", "united arab emirates", "uae", "spain", "italy", "brazil", "mexico", "netherlands",
    "ireland", "switzerland", "south korea", "indonesia", "philippines", "thailand", "bharat",
}

NAME_HINTS: dict[SemanticType, set[str]] = {
    SemanticType.ID: {"identifier", "key", "uid", "uuid", "code", "number", "ref", "reference"},
    SemanticType.NAME: {"name", "fullname"},
    SemanticType.EMAIL: {"email", "mail"},
    SemanticType.PHONE: {"phone", "telephone", "mobile", "cell", "contact"},
    SemanticType.DATE: {"date", "day", "dob", "birth"},
    SemanticType.DATETIME: {"datetime", "timestamp", "time", "created", "updated", "modified"},
    SemanticType.CITY: {"city", "town", "municipality", "locality"},
    SemanticType.COUNTRY: {"country", "nation"},
    SemanticType.ZIPCODE: {"zipcode", "zip", "postal", "pincode", "postcode"},
    SemanticType.CURRENCY: {"amount", "price", "cost", "fare", "spent", "revenue", "tip", "tax", "fee", "payment", "salary", "income", "purchase", "surcharge", "toll"},
    SemanticType.CURRENCY_CODE: {"currency"},
    SemanticType.LATITUDE: {"latitude", "lat"},
    SemanticType.LONGITUDE: {"longitude", "lon", "lng"},
}


def _rate(values: list[str], pred) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if pred(v)) / len(values)


def infer_semantic_type(name: str, stats: dict[str, Any], row_count: int) -> tuple[SemanticType, float, dict[str, float]]:
    tokens = set(name_tokens(name))
    dt: DataType = stats["data_type"]
    sample = [s for s in (stats.get("raw_sample") or []) if s is not None and str(s).strip()]
    non_null = max(stats.get("non_null", 0), 1)
    uniqueness = stats.get("distinct", 0) / non_null
    scores: dict[SemanticType, float] = {}

    def hint(t: SemanticType) -> float:
        return 1.0 if tokens & NAME_HINTS.get(t, set()) else 0.0

    is_numeric = dt in (DataType.INTEGER, DataType.FLOAT)
    is_string = dt == DataType.STRING

    if dt == DataType.BOOLEAN:
        scores[SemanticType.BOOLEAN] = 1.0
    if dt in (DataType.DATE, DataType.DATETIME):
        scores[SemanticType.DATETIME if dt == DataType.DATETIME else SemanticType.DATE] = 0.95 + 0.05 * hint(SemanticType.DATE)

    if is_string and sample:
        email = _rate(sample, lambda v: bool(EMAIL_RE.match(v)))
        scores[SemanticType.EMAIL] = 0.8 * email + 0.2 * hint(SemanticType.EMAIL) if email > 0.5 else 0.3 * hint(SemanticType.EMAIL)

        parse_rate, _fmt, has_time = detect_date_formats(sample[:300])
        if parse_rate > 0.6:
            t = SemanticType.DATETIME if has_time else SemanticType.DATE
            scores[t] = max(scores.get(t, 0), 0.75 * parse_rate + 0.25 * max(hint(SemanticType.DATE), hint(SemanticType.DATETIME)))

        lower = [v.strip().lower() for v in sample]
        city = _rate(lower, lambda v: v in CITY_GAZETTEER)
        scores[SemanticType.CITY] = 0.6 * min(1.0, city * 1.5) + 0.4 * hint(SemanticType.CITY)
        country = _rate(lower, lambda v: v in COUNTRY_GAZETTEER)
        scores[SemanticType.COUNTRY] = 0.6 * min(1.0, country * 1.2) + 0.4 * hint(SemanticType.COUNTRY)

        codes = iso_currency_codes()
        code_rate = _rate(sample, lambda v: v.strip().upper() in codes)
        if code_rate > 0.8:
            scores[SemanticType.CURRENCY_CODE] = 0.8 * code_rate + 0.2 * hint(SemanticType.CURRENCY_CODE)
        money_rate = _rate(sample, lambda v: bool(CURRENCY_TEXT_RE.match(v)) and parse_currency(v)[1] is not None)
        if money_rate > 0.3:
            scores[SemanticType.CURRENCY] = 0.7 * money_rate + 0.3 * hint(SemanticType.CURRENCY)

        phone = _rate(sample, lambda v: bool(PHONE_RE.match(v)) and sum(ch.isdigit() for ch in v) >= 7)
        if phone > 0.6:
            scores[SemanticType.PHONE] = 0.6 * phone + 0.4 * hint(SemanticType.PHONE)

        person = _rate(sample, lambda v: bool(PERSON_NAME_RE.match(v.strip())))
        scores[SemanticType.NAME] = 0.55 * person + 0.45 * hint(SemanticType.NAME)

        alnum_id = _rate(sample, lambda v: bool(ALNUM_ID_RE.match(v.strip())))
        if alnum_id > 0.8:
            scores[SemanticType.ID] = 0.45 * alnum_id + 0.35 * min(1.0, uniqueness * 1.05) + 0.2 * hint(SemanticType.ID)
        # token identifiers: one token, letters *and* digits, near-constant length. This is a value signature, so it
        # holds for foreign-key columns too (low uniqueness), and it outranks the category / free-text fallbacks.
        stripped = [v.strip() for v in sample]
        token = _rate(stripped, lambda v: bool(TOKEN_ID_RE.match(v)) and any(ch.isdigit() for ch in v) and any(ch.isalpha() for ch in v))
        lengths = [len(v) for v in stripped if v]
        if token > 0.9 and lengths and max(lengths) - min(lengths) <= 4:
            scores[SemanticType.ID] = max(scores.get(SemanticType.ID, 0.0), 0.6 + 0.25 * token + 0.15 * hint(SemanticType.ID))

        zipr = _rate(sample, lambda v: bool(ZIP_RE.match(v.strip())))
        if zipr > 0.8 and hint(SemanticType.ZIPCODE):
            scores[SemanticType.ZIPCODE] = 0.5 * zipr + 0.5

        avg_len = stats.get("avg_length") or 0
        if avg_len > 40:
            scores[SemanticType.FREE_TEXT] = min(1.0, 0.5 + avg_len / 200)
        if stats.get("distinct", 0) <= 60 or uniqueness < 0.05:
            scores[SemanticType.CATEGORY] = max(scores.get(SemanticType.CATEGORY, 0), 0.55)

    measure_tokens = {"population", "count", "total", "sum", "percent", "pct", "ratio", "rate", "change", "median", "mean", "average", "income", "distance", "duration", "quantity", "age", "score", "number_of"}
    if is_numeric:
        mn, mx = stats.get("min"), stats.get("max")
        integral = (stats.get("integral_rate") or 0) >= 0.999
        if hint(SemanticType.LATITUDE) and mn is not None and -90 <= mn and mx <= 90:
            scores[SemanticType.LATITUDE] = 0.95
        if hint(SemanticType.LONGITUDE) and mn is not None and -180 <= mn and mx <= 180:
            scores[SemanticType.LONGITUDE] = 0.95
        if hint(SemanticType.ZIPCODE) and integral:
            scores[SemanticType.ZIPCODE] = 0.9
        if hint(SemanticType.CURRENCY) and not hint(SemanticType.ID):
            scores[SemanticType.CURRENCY] = 0.7 + 0.2 * hint(SemanticType.CURRENCY)
        if integral and not (tokens & measure_tokens):
            id_score = 0.55 * hint(SemanticType.ID) + 0.3 * min(1.0, uniqueness) + (0.15 if (mn is not None and mn >= 0) else 0)
            if hint(SemanticType.ID) or uniqueness >= 0.98:
                scores[SemanticType.ID] = id_score
            if not hint(SemanticType.ZIPCODE) and stats.get("distinct", 0) <= 12 and not hint(SemanticType.ID):
                scores[SemanticType.CATEGORY] = 0.5
        if hint(SemanticType.PHONE):
            scores[SemanticType.PHONE] = 0.7
        scores.setdefault(SemanticType.NUMERIC, 0.4)

    if not scores:
        return SemanticType.UNKNOWN, 0.0, {}
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] < 0.35:
        fallback = SemanticType.NUMERIC if is_numeric else (SemanticType.FREE_TEXT if (stats.get("avg_length") or 0) > 25 else SemanticType.UNKNOWN)
        return fallback, round(best[1], 3), {k.value: round(v, 3) for k, v in scores.items()}
    return best[0], round(min(1.0, best[1]), 3), {k.value: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:5]}
