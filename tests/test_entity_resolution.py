import itertools
from collections import defaultdict

from backend.core.models import SemanticType as S
from backend.entity_resolution.blocking import generate_candidates
from backend.entity_resolution.clustering import connected_components_clusters, correlation_clusters
from backend.entity_resolution.comparators import name_compare, phone_compare
from backend.entity_resolution.resolver import FieldSpec, resolve
from backend.ingestion.loader import ingest_file


def test_name_comparator_levels():
    assert name_compare("Rahul Sharma", "Rahul Sharma")[0] == "exact"
    assert name_compare("Rahul Sharma", "RAHUL K SHARMA")[0] == "high"
    assert name_compare("Rahul Sharma", "R. Sharma")[0] == "medium"
    assert name_compare("Rahul Sharma", "Rohan Sharma")[0] == "different"
    assert name_compare(None, "x")[0] == "null"
    assert phone_compare("+91-98765-43210", "9876543210")[0] == "exact"


def test_blocking_reduces_comparisons():
    records = {(0, i): {"name": f"Person{i} Last{i % 50}", "email": f"p{i}@x.com"} for i in range(500)}
    records.update({(1, i): {"name": f"Person{i} Last{i % 50}", "email": f"p{i}@x.com"} for i in range(500)})
    pairs, stats = generate_candidates(records, [("name", S.NAME, "basic"), ("email", S.EMAIL, "basic")], allow_within=set(), max_block_size=1000, window=3, schemes=["standard", "token", "sorted_neighborhood"])
    assert ((0, 7), (1, 7)) in pairs
    assert stats.reduction_ratio > 0.95


def test_correlation_clustering_prevents_transitive_chaining():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): 0.97, ("b", "c"): 0.96, ("a", "c"): 0.001}
    cc = connected_components_clusters(nodes, edges, 0.9)
    corr = correlation_clusters(nodes, edges, threshold=0.9)
    assert any(len(c) == 3 for c in cc), "baseline chains a~b~c"
    assert not any({"a", "c"} <= c for c in corr), "correlation clustering must not merge an explicit non-match"


def test_resolution_accuracy_on_sample(workspace, config, sample_files, ground_truth):
    a = ingest_file(workspace, sample_files[0])
    b = ingest_file(workspace, sample_files[1])
    fields = [
        FieldSpec("name", S.NAME, {a.dataset_id: "customer_name", b.dataset_id: "full_name"}),
        FieldSpec("city", S.CITY, {a.dataset_id: "city", b.dataset_id: "location"}, "semantic:CITY"),
        FieldSpec("email", S.EMAIL, {a.dataset_id: "email", b.dataset_id: "email_address"}),
        FieldSpec("phone", S.PHONE, {a.dataset_id: "phone", b.dataset_id: "phone_number"}, "digits"),
    ]
    res = resolve({a.dataset_id: a.storage_path, b.dataset_id: b.storage_path}, fields, config, auto_threshold=0.9)
    gt = ground_truth["entities"]

    def pairs(groups):
        return {tuple(sorted(p)) for g in groups for p in itertools.combinations(g, 2)}

    truth_groups = defaultdict(list)
    for ds, key in ((a.dataset_id, "sample_customers"), (b.dataset_id, "sample_customer_master")):
        for row, ent in enumerate(gt[key]):
            truth_groups[ent].append((ds, row))
    pred_groups = defaultdict(list)
    for rec, ent in res.assignment.items():
        pred_groups[ent].append(rec)
    T, P = pairs(truth_groups.values()), pairs(pred_groups.values())
    precision, recall = len(P & T) / len(P), len(P & T) / len(T)
    assert precision >= 0.95 and recall >= 0.95, (precision, recall)
    assert res.stats["reduction_ratio"] > 0.9
    assert sum(c.size for c in res.clusters) == a.row_count + b.row_count
    assert res.model.to_dict()["fields"]["email"]["exact"]["bayes_factor"] > 10
