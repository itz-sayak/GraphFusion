"""Download every public source used by the NYC mobility integration demo.

    python scripts/download_demo_data.py            # 500K-trip subset
    python scripts/download_demo_data.py --rows 100000

Sources (all public):
  1. NYC TLC Yellow Taxi Trip Records           Parquet   (Trip Operations System)
  2. NYC TLC Taxi Zone Lookup                    CSV       (Location Master System)
  3. NYC OpenData NTA census demographics        JSON/API  (City Analytics System)
  4. US Census ACS 5-year county estimates       JSON/API  (External Demographics System)
"""
from __future__ import annotations

import argparse

import download_census_data
import download_nyc_opendata
import download_tlc_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--month", default="2024-01")
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("[1/3] NYC TLC trips + zone lookup")
    download_tlc_data.main(a.month, a.rows, 42, a.force)
    print("[2/3] NYC OpenData NTA demographics")
    download_nyc_opendata.main(a.force)
    print("[3/3] US Census ACS county data")
    download_census_data.main(2023, a.force)
    print("\nDone. Load them in the UI with 'NYC demo', or run: python scripts/run_demo.py --nyc")


if __name__ == "__main__":
    main()
