from datetime import datetime

import pytest

from backend.conflicts.strategies import Candidate, estimate_source_accuracy, resolve


def cands():
    return [
        Candidate("Mumbai", "mumbai", "crm", "city", 1, 1.0, datetime(2024, 1, 1), 0),
        Candidate("Pune", "pune", "mdm", "location", 7, 0.93, datetime(2025, 6, 1), 1),
        Candidate("Mumbai", "mumbai", "crm", "city", 9, 0.95, datetime(2023, 1, 1), 0),
    ]


@pytest.mark.parametrize(
    "strategy,expected,status",
    [
        ("majority_vote", "Mumbai", "flagged"),
        ("prefer_source", "Mumbai", "flagged"),
        ("prefer_latest", "Pune", "flagged"),
        ("prefer_non_null", "Mumbai", "flagged"),
        ("highest_confidence", "Mumbai", "flagged"),
        ("manual_review", None, "manual_review"),
    ],
)
def test_strategies(strategy, expected, status):
    res = resolve(cands(), strategy)
    assert res.value == expected and res.status == status
    assert len(res.candidates) == 3, "all candidates are retained for the report"


def test_keep_all_and_agreement():
    assert resolve(cands(), "keep_all").value == '["Mumbai", "Pune"]'
    same = [Candidate("Bangalore", "bengaluru", "a", "c", 1), Candidate("Bengaluru", "bengaluru", "b", "c", 2, priority=1)]
    res = resolve(same, "majority_vote")
    assert res.status == "resolved" and res.value == "Bangalore"


def test_prefer_latest_without_timestamps_falls_back():
    c = [Candidate("A", "a", "x", "c", 1), Candidate("B", "b", "y", "c", 2, priority=1), Candidate("A", "a", "y", "c", 3, priority=1)]
    res = resolve(c, "prefer_latest")
    assert res.value == "A" and "fell back" in res.reason


def test_source_accuracy_truth_discovery():
    sets = []
    for i in range(20):
        truth = f"v{i}"
        sets.append([Candidate(truth, truth, "good", "c", i), Candidate(truth, truth, "ok", "c", i, priority=1), Candidate("junk", "junk", "bad", "c", i, priority=2)])
    acc = estimate_source_accuracy(sets)
    assert acc["good"] > 0.9 and acc["bad"] < 0.1
    tie = [Candidate("x", "x", "bad", "c", 1), Candidate("y", "y", "good", "c", 1, priority=1)]
    assert resolve(tie, "source_accuracy_vote", acc).value == "y"


def test_merge_conflicts_are_recorded_not_overwritten(merged_session):
    c = merged_session.conflicts(limit=1000)
    assert c["total"] > 0
    for conflict in c["conflicts"]:
        values = {str(x["value"]) for x in conflict["candidates"]}
        assert len(values) >= 2
        assert sum(x["selected"] for x in conflict["candidates"]) <= 1
        assert all(x["source_dataset"] and x["source_row"] is not None for x in conflict["candidates"])
