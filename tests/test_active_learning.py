from backend.entity_resolution.clustering import correlation_clusters
from backend.entity_resolution.fellegi_sunter import fit_label_calibration
from backend.entity_resolution.resolver import ERResult, review_queue
from backend.entity_resolution.blocking import BlockingStats
from backend.entity_resolution.fellegi_sunter import FSModel


def _result(probs: dict, assignment: dict) -> ERResult:
    return ERResult(pairs=[], clusters=[], assignment=assignment, model=FSModel(m={}, u={}, prior=0.1), blocking=BlockingStats(), pair_probabilities=probs)


def test_constraints_override_evidence():
    nodes = ["a", "b", "c"]
    edges = {("a", "b"): 0.99, ("b", "c"): 0.01}
    merged = correlation_clusters(nodes, edges, threshold=0.9)
    assert any({"a", "b"} <= c for c in merged)
    split = correlation_clusters(nodes, edges, threshold=0.9, constraints={("a", "b"): False, ("b", "c"): True})
    assert not any({"a", "b"} <= c for c in split), "cannot-link must separate a strong pair"
    assert any({"b", "c"} <= c for c in split), "must-link must join a weak pair"


def test_review_queue_ranks_by_margin_around_threshold():
    a, b, c, d = ("x", 0), ("y", 0), ("y", 1), ("y", 2)
    probs = {(a, b): 0.9, (a, c): 0.5, (a, d): 0.02}
    q = review_queue(_result(probs, {a: "E1", b: "E2", c: "E3", d: "E4"}), limit=3, threshold=0.9)
    assert [x["probability"] for x in q][:2] == [0.9, 0.5], "the pair at the threshold is the most informative"
    assert q[0]["disagreement"], "p ≥ t but not merged is a disagreement"
    assert not review_queue(_result(probs, {}), exclude={(a, b), (c, a), (a, d)}, threshold=0.9)


def test_label_calibration_moves_probabilities_toward_labels():
    import math

    weights = [x / 2 for x in range(-20, 21)]
    # labels say pairs with W ≥ 6 bits are matches and below are not: stricter than σ(ln2·W)
    labels_w = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    labels_y = [False, False, False, False, True, True, True, True]
    a, b = fit_label_calibration(labels_w, labels_y, weights, label_weight=20.0)
    p = lambda w: 1 / (1 + math.exp(-(a * w + b)))  # noqa: E731
    assert p(3.0) < 1 / (1 + math.exp(-math.log(2) * 3.0)), "labelled non-matches lower their probability"
    assert fit_label_calibration([1.0, 2.0], [True, True]) is None, "needs both classes"


def test_session_review_and_label_flow(merged_session):
    s = merged_session
    queue = s.entity_review_queue(limit=10)
    assert queue["pairs"], "the sample merge has uncertain pairs"
    samplings = {p["sampling"] for p in queue["pairs"]}
    assert samplings <= {"uncertainty", "random"} and "uncertainty" in samplings
    top = queue["pairs"][0]
    assert top["left"]["values"] and set(top["left"]["values"]) == set(top["right"]["values"])
    # label the most uncertain pair the opposite of what the merge decided, then re-merge
    decision = "non_match" if top["same_entity_now"] else "match"
    r = s.label_entity_link(top["left"], top["right"], decision)
    assert r["labels"] == 1
    s.execute()
    er = next(iter(s.current_merge().er_results.values()))
    left, right = (top["left"]["dataset"], top["left"]["row"]), (top["right"]["dataset"], top["right"]["row"])
    same = er.assignment[left] == er.assignment[right]
    assert same == (decision == "match"), "the label is a hard constraint on the next merge"
    assert er.stats["user_labels"] == 1
