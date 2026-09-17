"""Shared fixtures. Sample data is generated fresh (small, seeded) into a temp dir,
so tests never depend on or modify the committed data/ directory."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DFG_LLM_PROVIDER", "mock")
os.environ.setdefault("DFG_LOG_LEVEL", "WARNING")

from backend.config import load_config  # noqa: E402
from backend.session.state import Session  # noqa: E402
from backend.storage.workspace import Workspace  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    import make_sample_data

    out = tmp_path_factory.mktemp("sample")
    make_sample_data.generate(out, n_entities=300, n_sales=1200, seed=11)
    return out


@pytest.fixture(scope="session")
def ground_truth(sample_dir) -> dict:
    return json.loads((sample_dir / "ground_truth.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_files(sample_dir) -> list[Path]:
    return [sample_dir / "sample_customers.csv", sample_dir / "sample_customer_master.json", sample_dir / "sample_sales.parquet"]


@pytest.fixture()
def config() -> dict:
    return load_config()


@pytest.fixture()
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "ws", "test")


@pytest.fixture(scope="module")
def merged_session(tmp_path_factory, sample_files) -> Session:
    """A session that has loaded, discovered and merged the sample data (shared per module)."""
    s = Session(workspace_root=tmp_path_factory.mktemp("session"))
    for f in sample_files:
        s.add_file(f)
    s.discover()
    s.execute()
    return s
