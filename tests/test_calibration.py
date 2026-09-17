import math
import random

import pytest

from backend.config import load_config
from backend.entity_resolution.fellegi_sunter import FSModel, prior_over_all_pairs, term_frequencies
from backend.merge.modes import effective_thresholds


def test_prior_over_all_pairs_recovers_known_rate_and_is_below_candidate_prior():
    rng = random.Random(3)
    total_pairs = 200_000
    true_matches = 400
    # candidates: every true match (strong evidence) plus look-alike non-matches (weak evidence)
    llrs = [rng.uniform(18, 24) for _ in range(true_matches)] + [rng.uniform(-6, 2) for _ in range(6_000)]
    candidate_prior = true_matches / len(llrs)
    lam = prior_over_all_pairs(llrs, total_pairs, start=candidate_prior)
    assert lam < candidate_prior
    assert lam == pytest.approx(true_matches / total_pairs, rel=0.1)


def test_calibrated_posterior_is_lower_for_weak_evidence():
    m = {"name": {"exact": 0.9, "high": 0.05, "medium": 0.03, "different": 0.02}}
    u = {"name": {"exact": 0.002, "high": 0.01, "medium": 0.05, "different": 0.938}}
    blocked = FSModel(m=m, u=u, prior=0.05)
    all_pairs = FSModel(m=m, u=u, prior=0.001)
    p_blocked, _ = blocked.probability({"name": "exact"})
    p_all, _ = all_pairs.probability({"name": "exact"})
    assert p_blocked > 0.95 > p_all > 0.2  # the same evidence is far weaker against the full population


def test_term_frequency_weakens_common_values():
    records = [{"name": "rahul sharma"} for _ in range(40)] + [{"name": f"person {i}"} for i in range(960)]
    freq = term_frequencies(records, "name", lambda v: v)
    m = {"name": {"exact": 0.9, "high": 0.05, "medium": 0.03, "different": 0.02}}
    u = {"name": {"exact": 0.003, "high": 0.01, "medium": 0.05, "different": 0.937}}
    model = FSModel(m=m, u=u, prior=0.001)
    common = model.llr({"name": "exact"}, {"name": freq["rahul sharma"]})
    rare = model.llr({"name": "exact"}, {"name": freq["person 7"]})
    assert common < rare
    assert common == pytest.approx(math.log2(0.9 / 0.04))


def test_merge_modes_use_record_link_thresholds():
    cfg = load_config()
    strict, balanced, permissive = (effective_thresholds(cfg, m) for m in ("strict", "balanced", "permissive"))
    assert strict["entity_merge"] > balanced["entity_merge"] > permissive["entity_merge"]
    assert balanced["auto_merge"] == 0.9 and balanced["entity_merge"] == 0.5  # schema thresholds unchanged
    over = effective_thresholds(cfg, "balanced", {"confidence_threshold": 0.95})
    assert over["entity_merge"] == 0.95  # "only merge above 95%" applies literally to calibrated probabilities
    no_uncertain = effective_thresholds(cfg, "balanced", {"merge_uncertain": False})
    assert no_uncertain["entity_merge"] >= cfg["thresholds"]["high"]


def test_resolver_reports_calibrated_prior(merged_session):
    er = next(iter(merged_session.current_merge().er_results.values()))
    prior = er.stats["prior"]
    assert prior["population"] == "all_pairs" and 0 < prior["lambda"] < 0.01
    assert all(0.0 <= p <= 1.0 for p in er.pair_probabilities.values())
