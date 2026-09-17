import math

import pytest

from backend.core.models import ColumnMatch, ColumnRef, ConfidenceBand, DatasetArtifact, DatasetRelationship, JoinKind, MatchEvidence, SourceType
from backend.graph import algorithms as ga
from backend.graph.schema_graph import IntegrationGraph, edge_cost
from backend.graph.serialize import react_flow_view, to_json


def _artifact(ds):
    return DatasetArtifact(dataset_id=ds, source_name=f"{ds}.csv", source_type=SourceType.CSV, source_uri=ds, storage_path=ds, row_count=10, column_count=2)


def _rel(a, b, c, kind=JoinKind.LOOKUP):
    return DatasetRelationship(left_dataset=a, right_dataset=b, confidence=c, band=ConfidenceBand.MEDIUM, join_kind=kind, evidence={"fact": a, "dimension": b, "coverage": 1.0, "coverage_basis": "distinct"})


def graph(edges, cost="neglog"):
    g = IntegrationGraph(cost)
    for d in {x for e in edges for x in e[:2]}:
        g.add_dataset(_artifact(d), None)
    g.add_dataset_relationships([_rel(a, b, c) for a, b, c in edges])
    return g


def test_edge_costs():
    assert edge_cost(1.0) == 0
    assert math.isclose(edge_cost(0.5), math.log(2))
    assert math.isclose(edge_cost(0.8, "linear"), 0.2)


def test_dijkstra_prefers_reliable_multi_hop_route():
    g = graph([("A", "B", 0.9), ("B", "C", 0.9), ("A", "C", 0.45)])
    route = ga.best_route(g, "A", "C")
    assert route["path"] == ["A", "B", "C"]
    assert math.isclose(route["reliability"], 0.81, abs_tol=1e-6)
    assert route["direct_confidence"] == 0.45 and route["uses_intermediate"]


def test_neglog_and_linear_costs_disagree_where_documented():
    # three 0.8 hops (reliability 0.512) versus a direct 0.45 edge
    g = graph([("A", "B", 0.8), ("B", "C", 0.8), ("C", "D", 0.8), ("A", "D", 0.45)])
    assert ga.best_route(g, "A", "D", cost_mode="neglog")["path"] == ["A", "B", "C", "D"]
    assert ga.best_route(g, "A", "D", cost_mode="linear")["path"] == ["A", "D"]


def test_maximum_spanning_tree_excludes_cycle_edge():
    g = graph([("A", "B", 0.95), ("B", "C", 0.9), ("A", "C", 0.6), ("D", "E", 0.8)])
    forest, excluded = ga.maximum_spanning_plan(g, 0.5)
    assert forest.number_of_edges() == 3
    assert [(e["left"], e["right"]) for e in excluded] in ([("A", "C")], [("C", "A")])
    assert excluded[0]["alternative_path"] in (["A", "B", "C"], ["C", "B", "A"])
    assert ga.integration_components(g, 0.5) == [["A", "B", "C"], ["D", "E"]]


def test_threshold_splits_components():
    g = graph([("A", "B", 0.95), ("B", "C", 0.4)])
    assert ga.integration_components(g, 0.5) == [["A", "B"], ["C"]]
    assert not ga.best_route(g, "A", "C", min_confidence=0.5)["found"]


def test_column_graph_and_serialisation():
    g = graph([("A", "B", 0.9)])
    m = ColumnMatch(left=ColumnRef(dataset_id="A", column="x"), right=ColumnRef(dataset_id="B", column="y"), score=0.9, band=ConfidenceBand.HIGH, evidence=MatchEvidence(), accepted=True)
    g.g.add_node("column:A::x", kind="column", dataset_id="A", column="x", label="x", semantic_type="ID", data_type="integer", uniqueness=1)
    g.g.add_node("column:B::y", kind="column", dataset_id="B", column="y", label="y", semantic_type="ID", data_type="integer", uniqueness=1)
    g.add_column_matches([m])
    assert ga.column_route(g, ("A", "x"), ("B", "y"))["found"]
    assert ga.column_groups([m]) == [{("A", "x"), ("B", "y")}]
    js = to_json(g)
    assert js["stats"]["edge_types"]["dataset_relationship"] == 1
    view = react_flow_view(g, "column")
    assert {n["type"] for n in view["nodes"]} == {"group", "column"} and len(view["edges"]) == 1


@pytest.mark.usefixtures("merged_session")
def test_session_graph_contains_expected_structure(merged_session):
    stats = merged_session.graph.stats()
    assert stats["node_kinds"]["dataset"] == 3
    assert stats["edge_types"]["contains"] >= 19
    route = merged_session.route("sample_sales", "sample_customer_master")
    assert route["path"] == ["sample_sales", "sample_customers", "sample_customer_master"]
