"""Download NYC OpenData neighbourhood demographics (JSON via the Socrata API, no key needed).

Dataset: "Census Demographics at the Neighborhood Tabulation Area (NTA) level"
(``rnsn-acs2``, NYC Department of City Planning) — population by NTA with
borough and county FIPS code. It plays the "City Analytics System" role: an
independently built JSON/API source whose schema shares no column names with
the TLC data (``geographic_area_borough`` vs ``Borough``).
"""
from __future__ import annotations

import argparse
import json

import httpx

from _common import PROCESSED, RAW

DATASET = "rnsn-acs2"
URL = f"https://data.cityofnewyork.us/resource/{DATASET}.json"


def fetch(page_size: int = 1000) -> list[dict]:
    records: list[dict] = []
    offset = 0
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        while True:
            r = client.get(URL, params={"$limit": page_size, "$offset": offset, "$order": ":id"})
            r.raise_for_status()
            batch = r.json()
            records.extend(batch)
            if len(batch) < page_size:
                return records
            offset += page_size


def main(force: bool) -> None:
    raw = RAW / "nta_demographics.json"
    if raw.exists() and not force:
        print(f"  cached  {raw}")
        records = json.loads(raw.read_text(encoding="utf-8"))
    else:
        print(f"NYC OpenData {DATASET}")
        records = fetch()
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"  fetched {len(records)} records")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "nta_demographics.json"
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"  {len(records)} NTA records → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
