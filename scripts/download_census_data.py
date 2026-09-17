"""US Census American Community Survey (ACS 5-year) county data for New York State.

Output format mirrors the official Census Data API JSON response (a header row
followed by value rows, all strings), e.g.::

    [["NAME","B01003_001E","B19013_001E",...,"state","county"],
     ["Kings County, New York","2631580","80263",...,"36","047"], ...]

Retrieval order:
1. **Official Census Data API** (``api.census.gov``) when ``CENSUS_API_KEY`` is set —
   the API now rejects keyless requests.
2. **Census Reporter API** (``api.censusreporter.org``, keyless), which republishes
   the same ACS tables; the response is converted to the official format.
3. The committed snapshot ``data/sample/census_acs_nyc_counties_snapshot.json``
   (real ACS values), so the demo is reproducible fully offline.

Variables: total population (B01003_001E), median household income
(B19013_001E), median gross rent (B25064_001E), persons below poverty
(B17001_002E) and poverty universe (B17001_001E), workers (B08301_001E) and
workers commuting by public transport (B08301_010E).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import httpx

from _common import PROCESSED, RAW, ROOT, SAMPLE

VARIABLES = ["B01003_001E", "B19013_001E", "B25064_001E", "B17001_002E", "B17001_001E", "B08301_001E", "B08301_010E"]
TABLES = sorted({v.split("_")[0] for v in VARIABLES})
SNAPSHOT = SAMPLE / "census_acs_nyc_counties_snapshot.json"


def official(key: str, year: int) -> list[list[str]]:
    params = {"get": ",".join(["NAME", *VARIABLES]), "for": "county:*", "in": "state:36", "key": key}
    r = httpx.get(f"https://api.census.gov/data/{year}/acs/acs5", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def census_reporter() -> tuple[list[list[str]], str]:
    r = httpx.get("https://api.censusreporter.org/1.0/data/show/latest", params={"table_ids": ",".join(TABLES), "geo_ids": "050|04000US36"}, timeout=90)
    r.raise_for_status()
    data = r.json()
    rows = [["NAME", *VARIABLES, "state", "county"]]
    for geoid, tables in sorted(data["data"].items()):
        name = data["geography"][geoid]["name"].replace(", NY", ", New York")
        values = []
        for v in VARIABLES:
            table, col = v.split("_")[0], v.split("_")[1][:3]
            est = tables[table]["estimate"].get(f"{table}{col}")
            values.append(None if est is None else str(int(est)))
        fips = geoid.split("US")[1]
        rows.append([name, *values, fips[:2], fips[2:]])
    return rows, data["release"]["name"]


def main(year: int, force: bool) -> None:
    out = PROCESSED / "census_acs_nyc_counties.json"
    raw = RAW / "census_acs_nyc_counties.json"
    key = os.environ.get("CENSUS_API_KEY") or _env_key()
    rows, source = None, None
    if key:
        try:
            rows, source = official(key, year), f"api.census.gov ACS {year} 5-year"
        except httpx.HTTPError as exc:
            print(f"  official Census API failed ({exc}); trying Census Reporter")
    if rows is None:
        try:
            rows, release = census_reporter()
            source = f"api.censusreporter.org ({release})"
        except httpx.HTTPError as exc:
            print(f"  Census Reporter failed ({exc}); using committed snapshot")
    if rows is None:
        shutil.copyfile(SNAPSHOT, out)
        print(f"  snapshot → {out}")
        return
    raw.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for p in (raw, out):
        p.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if force or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  {len(rows) - 1} New York counties from {source} → {out.relative_to(ROOT)}")


def _env_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("CENSUS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--force", action="store_true", help="also refresh the committed snapshot")
    a = ap.parse_args()
    main(a.year, a.force)
