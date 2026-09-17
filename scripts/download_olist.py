"""Download the Olist Brazilian e-commerce datasets (testing only).

Source: https://github.com/olist/work-at-olist-data/tree/master/datasets
Files land in data/raw/olist/ (git-ignored). Sizes are checked against the
GitHub API listing; files already present with the right size are skipped.

    python scripts/download_olist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "olist"
LISTING = "https://api.github.com/repos/olist/work-at-olist-data/contents/datasets"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        files = [f for f in client.get(LISTING).raise_for_status().json() if f["type"] == "file"]
        for f in files:
            dest = OUT / f["name"]
            if dest.exists() and dest.stat().st_size == f["size"]:
                print(f"  = {f['name']} ({f['size']:,} bytes, already present)")
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            with client.stream("GET", f["download_url"]) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
            if tmp.stat().st_size != f["size"]:
                print(f"  ✗ {f['name']}: expected {f['size']:,} bytes, got {tmp.stat().st_size:,}")
                return 1
            tmp.replace(dest)
            print(f"  ✔ {f['name']} ({f['size']:,} bytes)")
    print(f"{len(files)} files in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
