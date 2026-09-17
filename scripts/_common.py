"""Shared helpers for the download scripts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nyc"
PROCESSED = ROOT / "data" / "processed" / "nyc"
SAMPLE = ROOT / "data" / "sample"
sys.path.insert(0, str(ROOT))


def download(url: str, dest: Path, force: bool = False, timeout: float = 300.0, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  cached  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            started = time.perf_counter()
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  download {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            tmp.replace(dest)
            print(f"\r  fetched {dest.relative_to(ROOT)} ({done / 1e6:.1f} MB in {time.perf_counter() - started:.1f}s)")
            return dest
        except httpx.HTTPError as exc:
            print(f"\n  attempt {attempt} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    return dest
