import pytest

from backend.core.values import make_domain_normalizer, normalizers
from backend.ingestion.loader import ingest_file
from backend.matching.matcher import MatchStrategy, match_schemas
from backend.matching.name_sim import name_similarity
from backend.matching.normalize import name_tokens
from backend.profiling.profiler import profile_dataset


def test_name_normalisation():
    assert name_tokens("cust_no") == ["customer", "number"]
    assert name_tokens("customerIdentifier") == ["customer", "identifier"]
    assert name_tokens("PULocationID") == ["pickup", "location", "identifier"]
    assert name_tokens("tpep_pickup_datetime", ("tpep",)) == ["pickup", "datetime"]


@pytest.mark.parametrize("a,b", [("customer_id", "cust_id"), ("customer_id", "customerIdentifier"), ("customer_number", "cust_no"), ("email", "email_address")])
def test_name_similarity_strong(a, b):
    assert name_similarity(a, b) >= 0.8


def test_name_similarity_generic_tokens_do_not_dominate():
    assert name_similarity("customer_id", "product_id") < 0.75


def test_reference_value_maps():
    cities = make_domain_normalizer("indian_cities")
    assert cities("Bangalore") == cities("Bengaluru") == "bengaluru"
    assert normalizers()["semantic:CITY"]("New York City") == "new york"
    assert make_domain_normalizer("nyc_boroughs")("Kings County, NY") == "brooklyn"


@pytest.fixture()
def profiled(workspace, config, sample_files):
    profiles, sketches = {}, {}
    for f in sample_files:
        art = ingest_file(workspace, f)
        p, s = profile_dataset(art, config)
        profiles[art.dataset_id], sketches[art.dataset_id] = p, s
    return profiles, sketches


def _pairs(result):
    return {tuple(sorted((m.left.key, m.right.key))) for m in result.accepted()}


def test_full_matcher_on_sample(profiled, config, ground_truth):
    result = match_schemas(*profiled, config)
    predicted = _pairs(result)
    truth = {tuple(sorted(p)) for p in ground_truth["column_matches"]}
    tp = len(predicted & truth)
    assert tp / len(predicted) == 1.0, f"false positives: {predicted - truth}"
    assert tp / len(truth) >= 0.7
    city = next(m for m in result.accepted() if {m.left.column, m.right.column} == {"city", "location"})
    assert city.evidence.normalizer.startswith(("semantic:", "domain:")), "alias-aware value normaliser must be chosen"
    assert result.flooding["converged"]


def test_bipartite_alignment_enforces_one_to_one(profiled, config):
    no_align = match_schemas(*profiled, config, MatchStrategy(bipartite=False, flooding=False), min_accept=0.3)
    aligned = match_schemas(*profiled, config, MatchStrategy(flooding=False), min_accept=0.3)

    def fan_out(res):
        seen = {}
        for m in res.accepted():
            if m.relationship == "foreign_key_candidate":
                continue
            for side, other in ((m.left.key, m.right.dataset_id), (m.right.key, m.left.dataset_id)):
                seen[(side, other)] = seen.get((side, other), 0) + 1
        return max(seen.values(), default=0)

    assert fan_out(aligned) == 1
    assert fan_out(no_align) >= fan_out(aligned)


def test_evidence_is_complete(profiled, config):
    result = match_schemas(*profiled, config)
    m = result.accepted()[0]
    ev = m.evidence
    assert 0 <= m.score <= 1
    for field in ("name_similarity", "datatype_similarity", "semantic_similarity", "value_overlap", "distribution_similarity", "pattern_similarity", "cardinality_similarity"):
        assert 0 <= getattr(ev, field) <= 1


def test_weights_are_configurable(profiled, config):
    config["schema_matching"]["weights"] = {k: (1.0 if k == "name" else 0.0) for k in config["schema_matching"]["weights"]}
    name_only = match_schemas(*profiled, config, MatchStrategy(flooding=False, bipartite=False), min_accept=0.0)
    for m in name_only.matches:
        assert abs(m.score - m.evidence.name_similarity) < 1e-3 or m.evidence.notes
