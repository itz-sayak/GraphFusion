"""Do LLM-proposed, data-verified value crosswalks recover correspondences between different vocabularies?

Each scenario has two datasets whose linking column uses a different vocabulary for the same
real-world things, and no static value map covers it. A negative control pairs two unrelated
vocabularies that must not be matched. Column-match precision / recall against the planted
correspondence, with crosswalks off and on (the configured LLM proposes; the data verifies).

    python experiments/run_crosswalk.py
"""
from __future__ import annotations

import copy
import random
import tempfile
from pathlib import Path

import pandas as pd

from common import RESULTS, markdown_table, save_json

from backend.config import load_config
from backend.session.state import Session

STATES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL",
          "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
          "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
          "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
          "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
COUNTRIES = {"Germany": "DEU", "France": "FRA", "Spain": "ESP", "Italy": "ITA", "Portugal": "PRT", "Netherlands": "NLD", "Belgium": "BEL", "Austria": "AUT", "Switzerland": "CHE",
             "Poland": "POL", "Sweden": "SWE", "Norway": "NOR", "Denmark": "DNK", "Finland": "FIN", "Ireland": "IRL", "Greece": "GRC", "Japan": "JPN", "China": "CHN",
             "India": "IND", "Brazil": "BRA", "Argentina": "ARG", "Mexico": "MEX", "Canada": "CAN", "Australia": "AUS", "Egypt": "EGY", "Kenya": "KEN", "Nigeria": "NGA",
             "South Africa": "ZAF", "Turkey": "TUR", "Indonesia": "IDN"}
MONTHS = {m: m[:3] for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
COLOURS = ["crimson", "teal", "amber", "violet", "olive", "maroon", "navy", "coral", "ivory", "indigo"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def scenario(out: Path, name: str, seed: int) -> tuple[list[Path], set[frozenset[str]]]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    if name == "negative_control":
        a = pd.DataFrame({"product_line": [rng.choice(COLOURS) for _ in range(400)], "units": [rng.randint(1, 50) for _ in range(400)]})
        b = pd.DataFrame({"shift_day": [rng.choice(WEEKDAYS) for _ in range(300)], "hours": [rng.randint(1, 12) for _ in range(300)]})
        pa, pb = out / "orders.csv", out / "shifts.csv"
        a.to_csv(pa, index=False)
        b.to_csv(pb, index=False)
        return [pa, pb], set()
    vocab, (ca, cb), (ta, tb), subset = {
        "us_states": (STATES, ("state", "region_code"), ("population", "shipments"), 1.0),
        "countries": (COUNTRIES, ("country", "market"), ("gdp_rank", "visits"), 0.8),
        "months": (MONTHS, ("month", "period"), ("rainfall_mm", "orders"), 1.0),
    }[name]
    names = list(vocab)
    a = pd.DataFrame({ca: names, f"{ta}": [rng.randint(100, 10_000) for _ in names]})
    keep = rng.sample(names, max(3, int(len(names) * subset)))
    rows = [rng.choice(keep) for _ in range(len(names) * 20)]
    b = pd.DataFrame({cb: [vocab[r] for r in rows], f"{tb}": [rng.randint(1, 900) for _ in rows]})
    pa, pb = out / f"{name}_reference.csv", out / f"{name}_facts.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    return [pa, pb], {frozenset((f"{pa.stem}::{ca}", f"{pb.stem}::{cb}"))}


def run(name: str, enabled: bool, seed: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="dfg_cw_"))
    files, truth = scenario(tmp / "data", name, seed)
    cfg = copy.deepcopy(load_config())
    cfg["schema_matching"]["crosswalk"]["enabled"] = enabled
    s = Session(workspace_root=tmp / "ws", config=cfg)
    for f in files:
        s.add_file(f)
    s.discover()
    accepted = {frozenset((m.left.key, m.right.key)) for m in s.matching.matches if m.accepted}
    # only the vocabulary columns are in question; measure on pairs that involve them
    vocab_cols = {k for p in truth for k in p} if truth else {f"{f.stem}::{c}" for f in files for c in pd.read_csv(f, nrows=0).columns if c in ("product_line", "shift_day")}
    relevant = {p for p in accepted if p & vocab_cols}
    tp = len(relevant & truth)
    cws = [c for c in s.crosswalks]
    return {"scenario": name, "crosswalk": "on" if enabled else "off", "truth": len(truth), "predicted": len(relevant), "tp": tp,
            "recall": tp / len(truth) if truth else None, "false_links": len(relevant - truth),
            "crosswalks_proposed": len(cws), "crosswalks_verified": sum(1 for c in cws if c["accepted"]),
            "relationship": next((f"{r.join_kind.value} {r.confidence:.2f}" for r in s.relationships), "none"),
            "details": [{k: c[k] for k in ("left", "right", "accepted", "reason", "coverage")} for c in cws]}


def main() -> None:
    rows = []
    for i, name in enumerate(["us_states", "countries", "months", "negative_control"]):
        for enabled in (False, True):
            r = run(name, enabled, seed=100 + i)
            rows.append(r)
            print(f"  {name:17s} crosswalk {r['crosswalk']:3s}: recall {r['recall']}  false links {r['false_links']}  verified {r['crosswalks_verified']}/{r['crosswalks_proposed']}  relationship {r['relationship']}")
    save_json("crosswalk.json", {"rows": rows})
    md = ["# Value crosswalks: different vocabularies for the same things", "",
          "Column-match recall on the planted vocabulary correspondence (no static value map covers these vocabularies) and false links involving the vocabulary columns, with LLM-proposed crosswalks off/on. The negative control must stay unlinked.", "",
          markdown_table(rows, ["scenario", "crosswalk", "truth", "tp", "false_links", "crosswalks_proposed", "crosswalks_verified", "relationship"],
                         ["scenario", "crosswalk", "true correspondences", "found", "false links", "crosswalks proposed", "verified", "dataset relationship"], {})]
    (RESULTS / "crosswalk.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
