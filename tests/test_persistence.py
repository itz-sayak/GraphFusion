import pytest
from fastapi.testclient import TestClient

from backend.core.errors import NotFoundError
from backend.session.state import SessionRegistry


def test_session_survives_a_restart(tmp_path, sample_files):
    reg = SessionRegistry(tmp_path)
    s = reg.get_or_create("restart-test")
    for f in sample_files:
        s.add_file(f)
    s.discover()
    s.set_preferences(conflict_strategy="prefer_latest")
    rec = s.execute(write_csv=False)
    pair = s.entity_review_queue(3)["pairs"][0]
    s.label_entity_link(pair["left"], pair["right"], "match")
    s.chat_history.append({"role": "user", "content": "merge them"})
    reg.persist("restart-test")

    fresh = SessionRegistry(tmp_path)  # a new process: nothing in memory
    assert any(x["session_id"] == "restart-test" for x in fresh.list())
    r = fresh.get_or_create("restart-test")
    assert set(r.artifacts) == set(s.artifacts) and r.preferences.conflict_strategy == "prefer_latest"
    assert r.current_merge().result.merge_id == rec.result.merge_id
    assert r.output_preview(limit=3)["rows"], "merge outputs are readable after restore"
    assert r.entity_labels == s.entity_labels and r.chat_history[-1]["content"] == "merge them"
    assert r.lock is not None
    r.execute(write_csv=False)  # the restored session keeps working


def test_sessions_are_private_to_their_owner(tmp_path):
    reg = SessionRegistry(tmp_path)
    s = reg.create(owner="alice")
    assert reg.get(s.session_id, owner="alice") is s
    with pytest.raises(NotFoundError):
        reg.get(s.session_id, owner="bob")
    with pytest.raises(NotFoundError):
        reg.get_or_create(s.session_id, owner="bob")
    assert [x["session_id"] for x in reg.list(owner="bob")] == []
    with pytest.raises(Exception):
        reg.get_or_create("../../etc")


def test_api_tokens_and_persistence_over_http(tmp_path, monkeypatch):
    from backend.api import deps
    from backend.config import get_settings

    monkeypatch.setenv("DFG_API_TOKENS", "alice:tok-a,bob:tok-b")
    monkeypatch.setenv("DFG_WORKSPACE", str(tmp_path))
    get_settings.cache_clear()
    deps.registry.cache_clear()
    try:
        from backend.main import create_app

        client = TestClient(create_app())
        assert client.get("/health").status_code == 200  # open
        assert client.get("/datasets").status_code == 401
        alice = {"Authorization": "Bearer tok-a"}
        sid = client.post("/sessions", headers=alice).json()["session_id"]
        h_alice = {**alice, "X-Session-ID": sid}
        assert client.post("/demo/sample", headers=h_alice).status_code == 200
        assert client.get("/datasets/sample_customers/profile", headers={"Authorization": "Bearer tok-b", "X-Session-ID": sid}).status_code == 404

        deps.registry.cache_clear()  # simulated backend restart
        client2 = TestClient(create_app())
        datasets = client2.get("/datasets", headers=h_alice).json()
        assert len(datasets) == 3, "datasets survive the restart"
        assert client2.get(f"/integration/export?session_id={sid}&token=tok-a").status_code in (409, 404)  # token via query for downloads
    finally:
        get_settings.cache_clear()
        deps.registry.cache_clear()
