"""Generate intentionally difficult sample datasets with ground truth.

Three independently-built "systems" describe overlapping customers:

    sample_customers.csv          CRM export     customer_id, customer_name, city, ...
    sample_customer_master.json   MDM system     cust_no (different id space!), full_name, location, ...
    sample_sales.parquet          POS system     txn_id, cust_id -> customers.customer_id, amount_spent ('₹1,200'), ...

Injected difficulties: renamed/abbreviated columns (cust_id / customer_id /
cust_no), city aliases (Bengaluru/Bangalore, New York/New York City, Bombay),
name variants (Rahul Sharma / Rahul K Sharma / RAHUL SHARMA), mixed date
formats, mixed currencies, duplicate customer accounts, orphan foreign keys,
missing values, conflicting attributes (people who moved city), and hard
negative columns (signup_date vs last_updated vs txn_date).

Ground truth (column correspondences + hidden entity ids per row) is written
to ``data/sample/ground_truth.json``. Fully deterministic (seeded).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Arjun", "Kavya", "Vikram", "Ananya", "Rohan", "Meera", "Karan", "Isha", "Aditya", "Pooja",
         "Siddharth", "Neha", "Manish", "Divya", "Suresh", "Lakshmi", "John", "Emily", "Michael", "Sarah", "David", "Jessica", "Daniel",
         "Laura", "James", "Olivia", "Ravi", "Anjali", "Nikhil", "Shreya", "Varun", "Tanvi", "Harish", "Nisha", "Gaurav", "Ritu"]
LAST = ["Sharma", "Iyer", "Patel", "Reddy", "Nair", "Gupta", "Mehta", "Rao", "Singh", "Kapoor", "Joshi", "Menon", "Das", "Bose",
        "Kulkarni", "Chopra", "Smith", "Johnson", "Brown", "Williams", "Miller", "Davis", "Garcia", "Wilson", "Taylor", "Anderson"]
MIDDLE = list("KSRMAPDVJN")
# canonical city -> (country, currency, variants seen in *other* systems)
CITIES = {
    "Mumbai": ("India", "INR", ["Bombay", "mumbai"]),
    "Bangalore": ("India", "INR", ["Bengaluru", "BANGALORE"]),
    "Delhi": ("India", "INR", ["New Delhi", "delhi"]),
    "Chennai": ("India", "INR", ["Madras"]),
    "Pune": ("India", "INR", ["Poona"]),
    "Hyderabad": ("India", "INR", ["hyderabad"]),
    "Kolkata": ("India", "INR", ["Calcutta"]),
    "New York": ("United States", "USD", ["New York City", "NYC"]),
    "San Francisco": ("United States", "USD", ["SF", "San Fran"]),
    "London": ("United Kingdom", "GBP", ["london"]),
}
SYMBOL = {"INR": "₹", "USD": "$", "GBP": "£", "EUR": "€"}


def fmt_date(d: date, rng: random.Random) -> str:
    style = rng.random()
    if style < 0.45:
        return d.isoformat()
    if style < 0.75:
        return d.strftime("%d/%m/%Y")
    if style < 0.9:
        return d.strftime("%d %b %Y")
    return d.strftime("%d-%m-%Y")


def name_variant(first: str, last: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return f"{first} {rng.choice(MIDDLE)} {last}"
    if r < 0.6:
        return f"{first} {last}".upper()
    if r < 0.8:
        return f"{first[0]}. {last}"
    return f"{first} {last}"


def phone_variant(digits: str, rng: random.Random, india: bool) -> str:
    r = rng.random()
    if india:
        if r < 0.5:
            return f"+91-{digits[:5]}-{digits[5:]}"
        return digits
    if r < 0.5:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"


def generate(out_dir: Path, n_entities: int = 600, n_sales: int = 3000, seed: int = 7,
             crm_email_missing: float = 0.05, crm_phone_missing: float = 0.08, mdm_email_missing: float = 0.1, mdm_phone_missing: float = 0.1) -> dict:
    """Missing-value rates are parameters so harder variants (e.g. for active learning) can be generated; defaults reproduce the committed data."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities = []
    used = set()
    for eid in range(n_entities):
        while True:
            first, last = rng.choice(FIRST), rng.choice(LAST)
            suffix = rng.randint(1, 999)
            email = f"{first.lower()}.{last.lower()}{suffix}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'corp.example'])}"
            if email not in used:
                used.add(email)
                break
        city = rng.choice(list(CITIES))
        india = CITIES[city][0] == "India"
        digits = ("9" if india else "4") + "".join(str(rng.randint(0, 9)) for _ in range(9))
        entities.append(
            {
                "entity": f"E{eid:05d}",
                "first": first,
                "last": last,
                "email": email,
                "city": city,
                "country": CITIES[city][0],
                "currency": CITIES[city][1],
                "digits": digits,
                "india": india,
                "signup": date(2021, 1, 1) + timedelta(days=rng.randint(0, 1200)),
            }
        )

    # ---------------------------------------------------------------- CRM (CSV)
    crm_rows, crm_entity, crm_ids = [], [], {}
    next_id = 1001
    for e in entities:
        if rng.random() < 0.85:
            crm_ids.setdefault(e["entity"], []).append(next_id)
            crm_rows.append(
                {
                    "customer_id": next_id,
                    "customer_name": f"{e['first']} {e['last']}",
                    "city": e["city"],
                    "email": e["email"] if rng.random() > crm_email_missing else "",
                    "phone": e["digits"] if rng.random() > crm_phone_missing else "",
                    "signup_date": fmt_date(e["signup"], rng),
                    "country": e["country"],
                }
            )
            crm_entity.append(e["entity"])
            next_id += 1
    # duplicate accounts: same person registered again with a variant
    dup_sources = rng.sample([e for e in entities if e["entity"] in crm_ids], k=int(0.06 * len(crm_rows)))
    for e in dup_sources:
        crm_ids[e["entity"]].append(next_id)
        crm_rows.append(
            {
                "customer_id": next_id,
                "customer_name": name_variant(e["first"], e["last"], rng),
                "city": rng.choice([e["city"], rng.choice(CITIES[e["city"]][2])]),
                "email": e["email"].upper() if rng.random() < 0.5 else e["email"],
                "phone": phone_variant(e["digits"], rng, e["india"]) if rng.random() < 0.7 else "",
                "signup_date": fmt_date(e["signup"] + timedelta(days=rng.randint(30, 400)), rng),
                "country": e["country"],
            }
        )
        crm_entity.append(e["entity"])
        next_id += 1
    # a few exact duplicate rows (export glitch)
    for idx in rng.sample(range(len(crm_rows)), k=12):
        crm_rows.append(dict(crm_rows[idx]))
        crm_entity.append(crm_entity[idx])
    order = list(range(len(crm_rows)))
    rng.shuffle(order)
    crm_rows = [crm_rows[i] for i in order]
    crm_entity = [crm_entity[i] for i in order]
    header = list(crm_rows[0])
    with open(out_dir / "sample_customers.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in crm_rows:
            fh.write(",".join(_csv_cell(r[h]) for h in header) + "\n")

    # ---------------------------------------------------------------- MDM (JSON)
    mdm_rows, mdm_entity, conflicts = [], [], 0
    for i, e in enumerate(entities):
        if rng.random() < 0.75:
            city = e["city"]
            moved = False
            if rng.random() < 0.06:
                city = rng.choice([c for c in CITIES if c != e["city"]])
                moved = True
                conflicts += 1
            elif rng.random() < 0.45:
                city = rng.choice(CITIES[city][2])
            mdm_rows.append(
                {
                    "cust_no": f"C-{50000 + i * 7 + rng.randint(0, 6)}",
                    "full_name": name_variant(e["first"], e["last"], rng) if rng.random() < 0.55 else f"{e['first']} {e['last']}",
                    "location": city,
                    "email_address": (e["email"] if rng.random() > 0.3 else e["email"].capitalize()) if rng.random() > mdm_email_missing else None,
                    "phone_number": phone_variant(e["digits"], rng, e["india"]) if rng.random() > mdm_phone_missing else None,
                    "last_updated": (datetime(2025, 1, 1) + timedelta(minutes=rng.randint(0, 500000))).isoformat(timespec="seconds"),
                    "segment": rng.choice(["retail", "premium", "enterprise"]),
                    "_moved": moved,
                }
            )
            mdm_entity.append(e["entity"])
    for r in mdm_rows:
        r.pop("_moved")
    with open(out_dir / "sample_customer_master.json", "w", encoding="utf-8") as fh:
        json.dump({"source": "customer-master-data-service", "exported_at": "2026-01-15T09:00:00", "records": mdm_rows}, fh, ensure_ascii=False, indent=1)

    # ---------------------------------------------------------------- POS (Parquet)
    all_ids = [(cid, eid) for eid, ids in crm_ids.items() for cid in ids]
    ent_by_id = {e["entity"]: e for e in entities}
    sales = {"txn_id": [], "cust_id": [], "amount_spent": [], "txn_date": [], "channel": []}
    sales_entity = []
    orphans = 0
    for t in range(n_sales):
        if rng.random() < 0.02:
            cid, eid = 90000 + t, None
            orphans += 1
            currency = "USD"
        else:
            cid, eid = rng.choice(all_ids)
            currency = ent_by_id[eid]["currency"]
        amount = round(rng.uniform(5, 400) * (80 if currency == "INR" else 1), 2)
        r = rng.random()
        if r < 0.4:
            amount_s = f"{SYMBOL[currency]}{amount:,.2f}"
        elif r < 0.7:
            amount_s = f"{currency} {amount:.2f}"
        else:
            amount_s = f"{amount:.2f} {currency}"
        sales["txn_id"].append(f"T{100000 + t}")
        sales["cust_id"].append(cid)
        sales["amount_spent"].append(amount_s)
        sales["txn_date"].append(fmt_date(date(2024, 1, 1) + timedelta(days=rng.randint(0, 600)), rng))
        sales["channel"].append(rng.choice(["online", "store", "mobile_app"]))
        sales_entity.append(eid)
    pq.write_table(pa.table(sales), out_dir / "sample_sales.parquet")

    gt = {
        "description": "Ground truth for the sample customer integration scenario",
        "datasets": {"sample_customers": "sample_customers.csv", "sample_customer_master": "sample_customer_master.json", "sample_sales": "sample_sales.parquet"},
        # unordered column correspondences (same real-world attribute)
        "column_matches": [
            ["sample_customers::customer_id", "sample_sales::cust_id"],
            ["sample_customers::customer_id", "sample_customer_master::cust_no"],
            ["sample_sales::cust_id", "sample_customer_master::cust_no"],
            ["sample_customers::customer_name", "sample_customer_master::full_name"],
            ["sample_customers::city", "sample_customer_master::location"],
            ["sample_customers::email", "sample_customer_master::email_address"],
            ["sample_customers::phone", "sample_customer_master::phone_number"],
        ],
        "join_keys": [["sample_sales::cust_id", "sample_customers::customer_id"]],
        "entities": {"sample_customers": crm_entity, "sample_customer_master": mdm_entity, "sample_sales": sales_entity},
        "stats": {
            "entities": n_entities,
            "crm_rows": len(crm_rows),
            "crm_duplicate_accounts": len(dup_sources),
            "crm_exact_duplicate_rows": 12,
            "mdm_rows": len(mdm_rows),
            "mdm_city_conflicts": conflicts,
            "sales_rows": n_sales,
            "sales_orphans": orphans,
        },
    }
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1)
    return gt["stats"]


def _csv_cell(v) -> str:
    s = "" if v is None else str(v)
    if any(ch in s for ch in ',"\n'):
        s = '"' + s.replace('"', '""') + '"'
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--entities", type=int, default=600)
    ap.add_argument("--sales", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    stats = generate(Path(args.out), args.entities, args.sales, args.seed)
    print(json.dumps(stats, indent=2))
