import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, sample_dir):
    os.environ["DFG_WORKSPACE"] = str(tmp_path_factory.mktemp("api_ws"))
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.main import create_app

    return TestClient(create_app())


@pytest.fixture(scope="module")
def session_headers(client, sample_files):
    files = [("files", (f.name, f.read_bytes())) for f in sample_files]
    r = client.post("/datasets/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body["errors"] and len(body["datasets"]) == 3
    return {"X-Session-ID": body["session_id"]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_dataset_endpoints(client, session_headers):
    assert len(client.get("/datasets", headers=session_headers).json()) == 3
    assert client.get("/datasets/sample_sales", headers=session_headers).json()["row_count"] == 1200
    prof = client.get("/datasets/sample_customers/profile", headers=session_headers).json()
    assert prof["duplicate_rows"] == 12
    assert client.get("/datasets/sample_customers/preview?limit=3", headers=session_headers).json()["rows"].__len__() == 3


def test_integration_flow(client, session_headers):
    d = client.post("/integration/discover", headers=session_headers).json()
    assert len(d["relationships"]) >= 2 and d["issues"]
    g = client.get("/integration/graph?level=dataset", headers=session_headers).json()
    assert len(g["nodes"]) == 3 and g["edges"]
    ex = client.get("/integration/explain/match", params={"left": "email", "right": "email_address"}, headers=session_headers).json()
    assert ex["factors"] and ex["accepted"]
    route = client.get("/integration/route", params={"source": "sample_sales", "target": "sample_customer_master"}, headers=session_headers).json()
    assert route["path"][0] == "sample_sales" and route["path"][-1] == "sample_customer_master"
    plan = client.post("/integration/plan", json={"mode": "balanced"}, headers=session_headers).json()
    assert plan["steps"] and plan["canonical_schema"]
    result = client.post("/integration/execute", json={"plan_id": plan["plan_id"]}, headers=session_headers).json()
    assert result["row_count"] == 1200 and result["validation"]["passed"]
    assert client.get("/integration/conflicts", headers=session_headers).json()["total"] >= 0
    assert client.get("/integration/provenance?column=email", headers=session_headers).json()["sources"]
    assert "DATA LINEAGE REPORT" in client.get("/integration/lineage-report", headers=session_headers).text
    exports = client.get("/integration/export", headers=session_headers).json()["files"]
    for f in ("unified_dataset.parquet", "unified_dataset.csv", "merge_report.json", "provenance.json", "conflicts.csv", "schema_mapping.json", "integration_graph.json"):
        assert f in exports
    dl = client.get("/integration/export?file=schema_mapping.json", headers=session_headers)
    assert dl.status_code == 200 and dl.json()["mappings"]


def test_entity_review_endpoints(client, session_headers):
    q = client.get("/integration/entities/review?limit=5", headers=session_headers).json()
    assert q["pairs"] and q["labels"] == 0
    p = q["pairs"][0]
    body = {"left": {"dataset": p["left"]["dataset"], "row": p["left"]["row"]}, "right": {"dataset": p["right"]["dataset"], "row": p["right"]["row"]},
            "decision": "match", "sampling": p["sampling"]}
    r = client.post("/integration/entities/label", json=body, headers=session_headers).json()
    assert r["labels"] == 1
    assert client.post("/integration/entities/label", json={**body, "decision": "maybe"}, headers=session_headers).status_code == 422


def test_chat_endpoint(client, session_headers):
    r = client.post("/chat", json={"message": "What percentage of rows were matched?", "session_id": session_headers["X-Session-ID"]})
    assert r.status_code == 200
    body = r.json()
    assert body["tool_calls"][0]["tool"] == "match_statistics" and body["suggestions"]


def test_errors_are_structured(client, session_headers):
    r = client.get("/datasets/does_not_exist/profile", headers=session_headers)
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = client.post("/integration/preferences", json={"mode": "chaotic"}, headers=session_headers)
    assert r.status_code == 422
    r = client.post("/datasets/load-path", json={"paths": ["../.env"]}, headers=session_headers)
    assert r.json()["errors"], "path traversal outside data/ must be rejected"
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_undo(client, session_headers):
    r = client.post("/integration/undo", headers=session_headers).json()
    assert r["undone"]
    assert client.get("/integration/export", headers=session_headers).status_code == 409
