"""Relationship inference on unseen, deliberately confusing schemas.

Generates star/snowflake scenarios (tests/test_schema_robustness.py::make_scenario) with
meaningless names and one of four identifier formats (UUID hex, UUID with dashes, prefixed
codes, integers with overlapping ranges), plus traps: disjoint identifiers with the same
format, look-alike date columns, a near-unique child reference, and sibling tables.
A scenario counts as correct only if the inferred relationships equal the ground truth
exactly, the near-unique reference is N:1 in the right direction, and no identifier pair
without shared values is accepted.

    python experiments/run_schema_robustness.py [--seeds 20]
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from common import RESULTS, ROOT, markdown_table, save_json

sys.path.insert(0, str(ROOT / "tests"))
from test_schema_robustness import FORMATS, make_scenario  # noqa: E402

from backend.session.state import Session  # noqa: E402

TRUTH = {frozenset(p) for p in (("items", "orders"), ("items", "prod"), ("pay", "orders"), ("rev", "orders"), ("orders", "cust"))}


def run_one(seed: int, fmt: str, config: dict | None = None) -> dict:
    d = Path(tempfile.mkdtemp(prefix="dfg_rob_"))
    names = make_scenario(d / "data", seed, fmt)
    ds = {v: k for k, v in names.items()}
    s = Session(workspace_root=d / "ws", config=copy.deepcopy(config) if config else None)
    for f in sorted((d / "data").glob("*.csv")):
        s.add_file(f)
    t0 = time.perf_counter()
    s.discover()
    found = {frozenset((ds[r.left_dataset], ds[r.right_dataset])): r for r in s.relationships}
    ro = found.get(frozenset(("rev", "orders")))
    bad_ids = sum(1 for m in s.matching.matches if m.accepted
                  and s.profiles[m.left.dataset_id].column(m.left.column).semantic_type.value == "ID"
                  and s.profiles[m.right.dataset_id].column(m.right.column).semantic_type.value == "ID"
                  and max(m.evidence.containment_left, m.evidence.containment_right) < 0.05)
    return {"seed": seed, "format": fmt, "missing": len(TRUTH - set(found)), "extra": len(set(found) - TRUTH),
            "n1_direction_ok": bool(ro and ro.join_kind.value == "lookup" and ds.get(ro.evidence.get("fact")) == "rev"),
            "disjoint_id_links": bad_ids, "discover_s": round(time.perf_counter() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--encoder", default=None, help="sentence-transformers model for the embedding signal (default: config)")
    args = ap.parse_args()
    config = None
    if args.encoder:
        from backend.config import load_config

        config = load_config()
        config["embeddings"] = {"backend": "sentence_transformers", "model": args.encoder}
    rows = [run_one(seed, fmt, config) for seed in range(10, 10 + args.seeds) for fmt in FORMATS]
    for r in rows:
        r["correct"] = r["missing"] == 0 and r["extra"] == 0 and r["n1_direction_ok"] and r["disjoint_id_links"] == 0
    summary = []
    for fmt in FORMATS:
        rs = [r for r in rows if r["format"] == fmt]
        summary.append({"format": fmt, "scenarios": len(rs), "correct": sum(r["correct"] for r in rs), "missed": sum(r["missing"] for r in rs),
                        "extra": sum(r["extra"] for r in rs), "wrong_direction": sum(not r["n1_direction_ok"] for r in rs),
                        "disjoint_id_links": sum(r["disjoint_id_links"] for r in rs), "discover_s": round(sum(r["discover_s"] for r in rs) / len(rs), 2)})
    suffix = "" if not args.encoder else "_" + args.encoder.split("/")[-1]
    save_json(f"schema_robustness{suffix}.json", {"encoder": args.encoder, "rows": rows, "summary": summary})
    md = ["# Relationship inference on unseen confusing schemas", "",
          f"{len(rows)} generated scenarios (6 tables, meaningless names, 5 true relationships each). Correct = exactly the true relationships, "
          "the near-unique child reference inferred as N:1 in the right direction, and no accepted link between identifiers that share no values.", "",
          markdown_table(summary, ["format", "scenarios", "correct", "missed", "extra", "wrong_direction", "disjoint_id_links", "discover_s"],
                         ["identifier format", "scenarios", "fully correct", "missed relationships", "extra relationships", "wrong N:1 direction", "false identifier links", "discovery (s)"],
                         {"discover_s": ".2f"})]
    if args.encoder:
        md[0] += f" (encoder: {args.encoder})"
    (RESULTS / f"schema_robustness{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
