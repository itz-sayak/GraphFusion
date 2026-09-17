import pandas as pd

from backend.config import load_config
from backend.matching.crosswalk import verify
from backend.session.state import Session, SessionRegistry

STATES = {"California": "CA", "Texas": "TX", "New York": "NY", "Florida": "FL", "Ohio": "OH", "Georgia": "GA", "Oregon": "OR", "Nevada": "NV"}


def _files(tmp_path):
    ref = pd.DataFrame({"state": list(STATES), "population": range(8)})
    facts = pd.DataFrame({"region_code": [list(STATES.values())[i % 8] for i in range(400)], "shipments": range(400)})
    (tmp_path / "d").mkdir()
    ref.to_csv(tmp_path / "d" / "states.csv", index=False)
    facts.to_csv(tmp_path / "d" / "shipments.csv", index=False)
    return [tmp_path / "d" / "states.csv", tmp_path / "d" / "shipments.csv"]


def fake_llm(req):
    lookup = {v.lower(): k.lower() for k, v in STATES.items()} | {k.lower(): v.lower() for k, v in STATES.items()}
    return {v: lookup.get(v) for v in req["left_values"]}


def test_verify_rejects_hallucinations_low_coverage_and_trivial_maps():
    cfg = load_config()["schema_matching"]["crosswalk"]
    left, right = ["ca", "tx", "ny", "fl"], ["california", "texas", "new york", "florida"]
    good = verify("a::code", "b::name", left, right, {"ca": "california", "tx": "texas", "ny": "new york", "fl": "florida"}, cfg)
    assert good.accepted and good.coverage == 1.0
    made_up = verify("a::code", "b::name", left, right, {"ca": "atlantis", "tx": "mordor", "ny": "new york"}, cfg)
    assert not made_up.accepted and "do not exist" in made_up.reason
    thin = verify("a::code", "b::name", left, right, {"ca": "california"}, cfg)
    assert not thin.accepted
    many_to_one = verify("a::code", "b::name", left, right, {"ca": "texas", "tx": "texas", "ny": "texas", "fl": "texas"}, cfg)
    assert not many_to_one.accepted and "one-to-one" in many_to_one.reason


def test_verified_crosswalk_enables_match_and_survives_restart(tmp_path):
    reg = SessionRegistry(tmp_path / "ws")
    s = reg.get_or_create("cw")
    s.crosswalk_proposer = fake_llm
    for f in _files(tmp_path):
        s.add_file(f)
    s.discover()
    pair = {"shipments::region_code", "states::state"}
    m = next(x for x in s.matching.matches if {x.left.key, x.right.key} == pair)
    assert m.accepted and m.evidence.normalizer.startswith("crosswalk:")
    assert any(r for r in s.relationships if {r.left_dataset, r.right_dataset} == {"shipments", "states"})
    s.execute(write_csv=False)
    reg.persist("cw")

    fresh = SessionRegistry(tmp_path / "ws").get_or_create("cw")  # restart: runtime normaliser must be re-registered
    fresh.discovery_stale = True
    fresh.execute(write_csv=False)
    assert fresh.current_merge().result.validation["passed"]


def test_no_llm_means_no_crosswalks(tmp_path):
    s = Session(workspace_root=tmp_path / "ws")  # tests run with the offline mock provider
    for f in _files(tmp_path):
        s.add_file(f)
    s.discover()
    assert s.crosswalks == []
