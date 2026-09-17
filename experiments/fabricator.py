"""Valentine-style fabricated schema-matching benchmark.

Following the fabrication methodology of Valentine (Koutras et al., ICDE'21),
each scenario derives three datasets from one base table by

* **vertical splits** — each dataset keeps a random subset of attributes (some shared);
* **horizontal splits** — each dataset keeps a random, partially overlapping row sample;
* **schema noise** — column renames of increasing severity;
* **instance noise** — typos, case/format changes, whitespace;
* **distractors** — extra plausible columns (codes, flags, scores) with no counterpart.

Ground truth is known exactly: two columns correspond iff they derive from the
same base attribute.

To avoid benchmarking the system against its own dictionaries, schema noise
is generated *independently* of the matcher's configuration: vowel dropping,
truncation, random prefixes/suffixes, case-style changes, a separate synonym
list (only partly overlapping the matcher's thesaurus), and — at the hard level —
fully opaque names (``attr_7``) where only instance evidence can help.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIRST = ["Aarav", "Maya", "Liam", "Zara", "Noah", "Ishaan", "Emma", "Kabir", "Olivia", "Arjun", "Sofia", "Vihaan", "Ava", "Rohan", "Mia", "Ethan", "Anika", "Lucas", "Diya", "Leo"]
LAST = ["Shah", "Kim", "Garcia", "Iyer", "Brown", "Khan", "Rossi", "Mehta", "Nguyen", "Silva", "Patel", "Fischer", "Das", "Moreau", "Tanaka", "Reddy"]
CITIES = ["Mumbai", "Delhi", "Pune", "London", "Paris", "Berlin", "Tokyo", "Chicago", "Toronto", "Madrid", "Dubai", "Singapore"]
COUNTRIES = {"Mumbai": "India", "Delhi": "India", "Pune": "India", "London": "United Kingdom", "Paris": "France", "Berlin": "Germany", "Tokyo": "Japan", "Chicago": "United States", "Toronto": "Canada", "Madrid": "Spain", "Dubai": "UAE", "Singapore": "Singapore"}

# independent synonym list for renames (deliberately not identical to config/abbreviations.yaml)
SYNONYMS = {
    "customer_id": ["client_key", "buyer_ref", "account_code"],
    "full_name": ["person", "display_name", "contact_label"],
    "email": ["mail_addr", "e_mail", "contact_mail"],
    "phone": ["mobile_no", "tel_number", "contact_phone"],
    "city": ["municipality", "town", "place"],
    "country": ["nation", "country_name", "ctry"],
    "signup_date": ["joined_on", "registration_day", "created"],
    "lifetime_value": ["ltv", "total_spend", "customer_value"],
    "segment": ["tier", "cohort", "customer_class"],
    "sku": ["item_code", "product_ref", "article_no"],
    "product_name": ["title", "item_description", "article"],
    "category": ["dept", "product_group", "family"],
    "price": ["unit_cost", "list_price", "retail_amount"],
    "weight_kg": ["mass", "wt", "shipping_weight"],
    "brand": ["maker", "manufacturer", "label"],
    "employee_id": ["staff_no", "worker_key", "badge"],
    "department": ["division", "org_unit", "team"],
    "salary": ["annual_pay", "compensation", "wage"],
    "hire_date": ["start_day", "joined", "onboarding_date"],
    "office_city": ["work_location", "site", "branch_town"],
    "flight_no": ["service_code", "flight_designator", "route_id"],
    "origin": ["from_airport", "dep_station", "source_port"],
    "destination": ["to_airport", "arr_station", "dest_port"],
    "departure_time": ["dep_ts", "scheduled_out", "leaves_at"],
    "distance_km": ["route_length", "km", "span"],
    "delay_min": ["lateness", "minutes_late", "slip"],
    "airline": ["carrier", "operator", "airline_name"],
}
DISTRACTORS = [("status_code", "int_small"), ("record_flag", "flag"), ("quality_score", "float"), ("batch_no", "int_small"), ("region_code", "code"), ("notes", "text")]


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    difficulty: str
    files: dict[str, Path]  # dataset_id -> file
    truth: set[tuple[str, str]]  # sorted pairs of "dataset::column"


def _base(domain: str, n: int, rng: random.Random) -> dict[str, list]:
    if domain == "customers":
        names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
        cities = [rng.choice(CITIES) for _ in range(n)]
        return {
            "customer_id": list(range(10_000, 10_000 + n)),
            "full_name": names,
            "email": [f"{nm.lower().replace(' ', '.')}{i}@mail.test" for i, nm in enumerate(names)],
            "phone": ["".join(rng.choice(string.digits) for _ in range(10)) for _ in range(n)],
            "city": cities,
            "country": [COUNTRIES[c] for c in cities],
            "signup_date": [(date(2020, 1, 1) + timedelta(days=rng.randint(0, 1500))).isoformat() for _ in range(n)],
            "lifetime_value": [round(rng.lognormvariate(6, 1), 2) for _ in range(n)],
            "segment": [rng.choice(["gold", "silver", "bronze"]) for _ in range(n)],
        }
    if domain == "products":
        cats = ["electronics", "garden", "toys", "kitchen", "books", "sports"]
        return {
            "sku": [f"SKU-{i:06d}" for i in range(n)],
            "product_name": [f"{rng.choice(['Pro', 'Ultra', 'Mini', 'Eco', 'Smart'])} {rng.choice(['Lamp', 'Kettle', 'Drone', 'Chair', 'Ball', 'Novel'])} {rng.randint(1, 99)}" for _ in range(n)],
            "category": [rng.choice(cats) for _ in range(n)],
            "price": [round(rng.uniform(3, 900), 2) for _ in range(n)],
            "weight_kg": [round(rng.uniform(0.1, 25), 3) for _ in range(n)],
            "brand": [rng.choice(["Acme", "Globex", "Initech", "Umbrella", "Stark"]) for _ in range(n)],
        }
    if domain == "employees":
        return {
            "employee_id": list(range(500, 500 + n)),
            "full_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
            "department": [rng.choice(["finance", "engineering", "sales", "legal", "hr"]) for _ in range(n)],
            "salary": [rng.randint(40, 250) * 1000 for _ in range(n)],
            "hire_date": [(date(2010, 1, 1) + timedelta(days=rng.randint(0, 5000))).isoformat() for _ in range(n)],
            "office_city": [rng.choice(CITIES) for _ in range(n)],
            "email": [f"emp{i}@corp.test" for i in range(n)],
        }
    # flights
    airports = ["BOM", "DEL", "LHR", "CDG", "FRA", "HND", "ORD", "YYZ", "MAD", "DXB", "SIN", "JFK"]
    return {
        "flight_no": [f"{rng.choice(['AI', 'BA', 'LH', 'EK'])}{rng.randint(100, 9999)}" for _ in range(n)],
        "origin": [rng.choice(airports) for _ in range(n)],
        "destination": [rng.choice(airports) for _ in range(n)],
        "departure_time": [f"2025-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}:00" for _ in range(n)],
        "distance_km": [rng.randint(300, 12000) for _ in range(n)],
        "delay_min": [max(0, int(rng.gauss(12, 25))) for _ in range(n)],
        "airline": [rng.choice(["Air India", "British Airways", "Lufthansa", "Emirates"]) for _ in range(n)],
    }


def _drop_vowels(s: str) -> str:
    return s[0] + "".join(ch for ch in s[1:] if ch not in "aeiou")


def _camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _rename(col: str, difficulty: str, rng: random.Random, idx: int) -> str:
    if difficulty == "easy":
        return rng.choice([col, col.upper(), _camel(col), col.replace("_", "")])
    if difficulty == "medium":
        ops = [
            lambda c: _camel(c),
            lambda c: "_".join(_drop_vowels(p) if len(p) > 4 else p for p in c.split("_")),
            lambda c: "_".join(p[:4] for p in c.split("_")),
            lambda c: f"src_{c}",
            lambda c: rng.choice(SYNONYMS.get(c, [c])),
        ]
        return rng.choice(ops)(col)
    # hard: synonyms, noisy synonyms, or fully opaque names
    r = rng.random()
    if r < 0.35:
        return f"attr_{idx}"
    syn = rng.choice(SYNONYMS.get(col, [col]))
    return _camel(syn) if r < 0.7 else "_".join(_drop_vowels(p) if len(p) > 3 else p for p in syn.split("_"))


def _noise_value(v, difficulty: str, rng: random.Random):
    if v is None or isinstance(v, (int, float)):
        return v
    p = {"easy": 0.02, "medium": 0.08, "hard": 0.15}[difficulty]
    if rng.random() > p:
        return v
    s = str(v)
    op = rng.random()
    if op < 0.35 and len(s) > 3:
        i = rng.randrange(len(s) - 1)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op < 0.6:
        return s.upper()
    if op < 0.8:
        return f" {s} "
    if len(s) == 10 and s[4] == "-":
        return f"{s[8:10]}/{s[5:7]}/{s[0:4]}"
    return s.lower()


def _distractor(kind: str, n: int, rng: random.Random) -> list:
    if kind == "int_small":
        return [rng.randint(1, 12) for _ in range(n)]
    if kind == "flag":
        return [rng.choice(["Y", "N"]) for _ in range(n)]
    if kind == "float":
        return [round(rng.random() * 100, 2) for _ in range(n)]
    if kind == "code":
        return [rng.choice(["NA", "EU", "APAC", "LATAM"]) for _ in range(n)]
    return [rng.choice(["ok", "check later", "vip", "", "duplicate?"]) for _ in range(n)]


def make_scenario(out_dir: Path, scenario_id: str, domain: str, difficulty: str, seed: int, n: int = 600) -> Scenario:
    rng = random.Random(seed)
    base = _base(domain, n, rng)
    attrs = list(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    columns_of: dict[str, dict[str, str]] = {}  # dataset -> new column -> base attribute
    overlap = {"easy": 0.9, "medium": 0.6, "hard": 0.35}[difficulty]
    shared_row_pool = rng.sample(range(n), int(n * overlap))
    for d in range(3):
        ds = f"{scenario_id}_d{d}"
        keep = [a for a in attrs if rng.random() < 0.75] or attrs[:2]
        rows = sorted(set(shared_row_pool) | set(rng.sample(range(n), int(n * (1 - overlap) * 0.8))))
        table: dict[str, list] = {}
        mapping: dict[str, str] = {}
        for idx, a in enumerate(keep):
            name = _rename(a, difficulty, rng, idx + d * 10)
            while name in table:
                name += "_x"
            table[name] = [_noise_value(base[a][r], difficulty, rng) for r in rows]
            mapping[name] = a
        for dname, kind in rng.sample(DISTRACTORS, k=rng.randint(1, 2)):
            if dname not in table:
                table[dname] = _distractor(kind, len(rows), rng)
        items = list(table.items())
        rng.shuffle(items)
        path = out_dir / f"{ds}.parquet"
        pq.write_table(pa.table(dict(items)), path)
        files[ds] = path
        columns_of[ds] = mapping
    truth = set()
    ds_ids = list(files)
    for i, a in enumerate(ds_ids):
        for b in ds_ids[i + 1:]:
            for ca, attr_a in columns_of[a].items():
                for cb, attr_b in columns_of[b].items():
                    if attr_a == attr_b:
                        truth.add(tuple(sorted((f"{a}::{ca}", f"{b}::{cb}"))))
    return Scenario(scenario_id, domain, difficulty, files, truth)


def make_benchmark(out_dir: Path, per_level: int = 8, seed: int = 2026) -> list[Scenario]:
    rng = random.Random(seed)
    domains = ["customers", "products", "employees", "flights"]
    scenarios = []
    for level in ("easy", "medium", "hard"):
        for i in range(per_level):
            sid = f"{level[0]}{i:02d}"
            scenarios.append(make_scenario(out_dir / sid, sid, domains[i % len(domains)], level, rng.randint(0, 10**9)))
    return scenarios
