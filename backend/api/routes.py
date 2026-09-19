"""REST endpoints.

Session selection: ``X-Session-ID`` header or ``session_id`` query parameter;
a session is created on first use and its id is echoed in the ``X-Session-ID``
response header. Handlers are synchronous and run in FastAPI's thread pool
because the work (DuckDB, entity resolution) is CPU-bound.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import Request, APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from backend import __version__
from backend.api import deps
from backend.api.schemas import (
    ChatRequest, ChatResponse, DiscoverRequest, EntityLabelRequest, EntityPreviewRequest, ExecuteRequest, LoadPathRequest, LoadUrlRequest,
    MappingDecisionRequest, PlanRequest, PreferencesRequest, SessionInfo, UploadResponse,
)
from backend.config import PROJECT_ROOT, get_settings
from backend.core.errors import DFGError, NotFoundError, ToolArgumentError
from backend.session.state import Session

router = APIRouter()
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
MAX_UPLOAD_BYTES = 4 * 1024**3
SAMPLE_FILES = ["sample/sample_customers.csv", "sample/sample_customer_master.json", "sample/sample_sales.parquet"]
NYC_FILES = ["processed/nyc/yellow_tripdata_sample.parquet", "processed/nyc/taxi_zone_lookup.csv", "processed/nyc/nta_demographics.json", "processed/nyc/census_acs_nyc_counties.json"]


# --------------------------------------------------------------------------- meta
@router.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    s = get_settings()
    return {"status": "ok", "version": __version__, "llm_provider": deps.agent().provider.describe(), "workspace": str(s.dfg_workspace)}


@router.get("/llm/usage", tags=["meta"])
def llm_usage() -> dict[str, Any]:
    """Client-side rate-limit accounting per model (last minute / last 24 h, limits, blocks)."""
    p = deps.agent().provider
    limiter = getattr(p, "limiter", None)
    if limiter is None:
        return {"provider": p.describe(), "models": []}
    return {"provider": p.describe(), "models": [limiter.usage(m) for m in getattr(p, "models", [p.model])]}


@router.get("/config", tags=["meta"])
def config(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    c = session.config
    return {"schema_matching_weights": c["schema_matching"]["weights"], "thresholds": c["thresholds"], "merge_modes": c["merge_modes"],
            "conflict_strategies": list(__import__("backend.merge.planner", fromlist=["CONFLICT_STRATEGIES"]).CONFLICT_STRATEGIES),
            "graph": c["graph"], "embeddings": c["embeddings"]}


@router.post("/sessions", response_model=SessionInfo, tags=["sessions"])
def create_session(request: Request) -> dict[str, Any]:
    s = deps.registry().create(owner=deps.current_user(request))
    request.state.session_id = s.session_id
    return s.snapshot()


@router.get("/sessions", tags=["sessions"])
def list_sessions(request: Request) -> list[dict[str, Any]]:
    """Your sessions, including ones persisted before a restart."""
    return deps.registry().list(owner=deps.current_user(request))


@router.get("/sessions/current", response_model=SessionInfo, tags=["sessions"])
def current_session(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.snapshot()


@router.delete("/sessions/{session_id}", tags=["sessions"])
def delete_session(session_id: str, request: Request) -> dict[str, str]:
    deps.registry().delete(session_id, owner=deps.current_user(request))
    return {"deleted": session_id}


# --------------------------------------------------------------------------- datasets
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@router.post("/datasets/upload", response_model=UploadResponse, tags=["datasets"])
def upload(files: list[UploadFile] = File(...), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    loaded, errors = [], []
    up_dir = session.workspace.root / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = _SAFE_NAME.sub("_", Path(f.filename or "upload").name)[:160] or "upload"
        dest = up_dir / name
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := f.file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ToolArgumentError(f"{name} exceeds the upload limit")
                    out.write(chunk)
            loaded.extend(a.summary() for a in session.add_files(dest, source_name=name))
        except DFGError as exc:
            errors.append({"file": name, "error": exc.message})
            dest.unlink(missing_ok=True)
    return {"session_id": session.session_id, "datasets": loaded, "errors": errors}


@router.post("/datasets/load-path", response_model=UploadResponse, tags=["datasets"])
def load_path(req: LoadPathRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Load files that already exist under the project's ``data/`` directory (path traversal is rejected)."""
    loaded, errors = [], []
    for p in req.paths:
        target = (DATA_ROOT / p).resolve()
        if DATA_ROOT not in target.parents or not target.is_file():
            errors.append({"file": p, "error": "path must be an existing file inside data/"})
            continue
        try:
            loaded.extend(a.summary() for a in session.add_files(target))
        except DFGError as exc:
            errors.append({"file": p, "error": exc.message})
    return {"session_id": session.session_id, "datasets": loaded, "errors": errors}


@router.post("/datasets/load-url", response_model=UploadResponse, tags=["datasets"])
def load_url(req: LoadUrlRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """OPTIONAL: ingest a JSON REST endpoint (Socrata ``$limit/$offset`` paging supported)."""
    if not req.url.startswith(("https://", "http://")):
        raise ToolArgumentError("url must be http(s)")
    art = session.add_rest(req.url, req.name, req.params, req.page_size)
    return {"session_id": session.session_id, "datasets": [art.summary()], "errors": []}


@router.post("/demo/{name}", response_model=UploadResponse, tags=["datasets"])
def load_demo(name: str, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    files = {"sample": SAMPLE_FILES, "nyc": NYC_FILES}.get(name)
    if files is None:
        raise NotFoundError("Unknown demo; use 'sample' or 'nyc'")
    missing = [f for f in files if not (DATA_ROOT / f).exists()]
    if missing:
        hint = "python scripts/make_sample_data.py" if name == "sample" else "python scripts/download_demo_data.py"
        raise NotFoundError(f"Demo data not found ({', '.join(missing)}). Run: {hint}")
    return load_path(LoadPathRequest(paths=files), session)


@router.get("/datasets", tags=["datasets"])
def list_datasets(session: Session = Depends(deps.get_session)) -> list[dict[str, Any]]:
    return session.list_datasets()


@router.get("/datasets/{dataset_id}", tags=["datasets"])
def get_dataset(dataset_id: str, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    art = session.artifacts.get(session.resolve_dataset_id(dataset_id))
    return {**art.summary(), "metadata": {k: v for k, v in art.metadata.items() if k not in ("raw_copy",)}, "checksum": art.checksum}  # type: ignore[union-attr]


@router.delete("/datasets/{dataset_id}", tags=["datasets"])
def delete_dataset(dataset_id: str, session: Session = Depends(deps.get_session)) -> dict[str, str]:
    session.remove_dataset(dataset_id)
    return {"removed": dataset_id}


@router.get("/datasets/{dataset_id}/profile", tags=["datasets"])
def get_profile(dataset_id: str, refresh: bool = False, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    ds = session.resolve_dataset_id(dataset_id)
    prof = session.profile(ds) if refresh or ds not in session.profiles else session.profiles[ds]
    return prof.model_dump(mode="json")


@router.get("/datasets/{dataset_id}/preview", tags=["datasets"])
def preview(dataset_id: str, limit: int = Query(20, ge=1, le=500), offset: int = Query(0, ge=0), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.preview(dataset_id, limit, offset)


# --------------------------------------------------------------------------- integration
@router.post("/integration/discover", tags=["integration"])
def discover(req: DiscoverRequest | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.discover(force=bool(req and req.force))


@router.get("/integration/graph", tags=["integration"])
def graph(level: str = Query("dataset", pattern="^(schema|dataset|column)$"), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.graph_view(level)


@router.get("/integration/graph/full", tags=["integration"])
def graph_full(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    from backend.graph.serialize import to_json

    session._ensure_discovered()
    return to_json(session.graph)  # type: ignore[arg-type]


@router.get("/integration/mappings", tags=["integration"])
def mappings(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    session._ensure_discovered()
    return session.schema_mapping_export()


@router.post("/integration/mappings/decision", tags=["integration"])
def mapping_decision(req: MappingDecisionRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.decide_match(req.left, req.right, req.decision)


@router.get("/integration/aggregate", tags=["results"])
def aggregate(measure: str | None = None, group_by: str | None = None, agg: str = Query("sum", pattern="^(sum|count|avg)$"), limit: int = Query(50, ge=1, le=1000),
              session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Aggregate the unified dataset with linkage uncertainty: point, expected value, 95% interval, certain-only."""
    return session.aggregate_with_uncertainty(measure, group_by, agg, limit)


@router.get("/integration/history", tags=["results"])
def history(entity_id: str | None = None, attribute: str | None = None, as_of: str | None = None, limit: int = Query(200, ge=1, le=5000),
            as_known_at: str | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Bitemporal attribute history: valid time (valid_from / valid_to, ``as_of``) and transaction time (``as_known_at``: what an earlier merge recorded)."""
    return session.entity_history(entity_id, attribute, as_of, limit, as_known_at)


@router.get("/integration/entities/review", tags=["integration"])
def entity_review(limit: int = Query(20, ge=1, le=200), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Record pairs of the current merge worth labelling (uncertainty + random calibration samples)."""
    return session.entity_review_queue(limit)


@router.post("/integration/entities/label", tags=["integration"])
def entity_label(req: EntityLabelRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Label a record pair as match / non_match; applied as a constraint on the next merge."""
    return session.label_entity_link(req.left.model_dump(), req.right.model_dump(), req.decision, req.sampling)


@router.post("/integration/matcher/retrain", tags=["integration"])
def retrain_matcher(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    """Refit the learned column matcher with this session's approve/reject decisions."""
    return session.retrain_matcher()


@router.get("/integration/explain/match", tags=["explainability"])
def explain_match(left: str, right: str, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.explain_match(left, right)


@router.get("/integration/explain/relationship", tags=["explainability"])
def explain_relationship(left: str, right: str, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.explain_relationship(left, right)


@router.get("/integration/route", tags=["explainability"])
def route(source: str, target: str, cost_mode: str | None = Query(None, pattern="^(neglog|linear)$"), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.route(source, target, cost_mode)


@router.post("/integration/entities/preview", tags=["integration"])
def entities_preview(req: EntityPreviewRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.entity_match_preview(req.left, req.right, req.limit)


@router.get("/integration/preferences", tags=["integration"])
def get_preferences(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.preferences_dict()


@router.post("/integration/preferences", tags=["integration"])
def set_preferences(req: PreferencesRequest, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.set_preferences(**req.model_dump(exclude_none=True))


@router.post("/integration/plan", tags=["integration"])
def plan(req: PlanRequest | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    req = req or PlanRequest()
    return session.make_plan(req.mode, req.conflict_strategy, req.datasets).model_dump(mode="json")


@router.get("/integration/plan", tags=["integration"])
def get_plan(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    if session.plan is None:
        raise NotFoundError("No plan has been generated yet")
    return session.plan.model_dump(mode="json")


@router.post("/integration/execute", tags=["integration"])
def execute(req: ExecuteRequest | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    req = req or ExecuteRequest()
    rec = session.execute(req.plan_id, write_csv=req.write_csv)
    return {**rec.result.model_dump(mode="json"), "plan": rec.plan.model_dump(mode="json"), "files": sorted(rec.files), "join_stats": rec.join_stats, "entity_groups": rec.group_stats}


@router.post("/integration/undo", tags=["integration"])
def undo(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.undo()


@router.get("/integration/output", tags=["results"])
def output(limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.output_preview(limit, offset)


@router.get("/integration/conflicts", tags=["results"])
def conflicts(limit: int = Query(100, ge=1, le=1000), attribute: str | None = None, entity_id: str | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.conflicts(limit, attribute, entity_id)


@router.get("/integration/duplicates", tags=["results"])
def duplicates(limit: int = Query(20, ge=1, le=200), session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.duplicates(limit)


@router.get("/integration/provenance", tags=["results"])
def provenance(column: str | None = None, session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    return session.provenance(column)


@router.get("/integration/lineage-report", response_class=PlainTextResponse, tags=["results"])
def lineage_report(session: Session = Depends(deps.get_session)) -> str:
    return session.lineage_report()


@router.get("/integration/quality", tags=["results"])
def quality(session: Session = Depends(deps.get_session)) -> dict[str, Any]:
    rec = session.current_merge()
    return {"quality_report": rec.result.quality_report, "validation": rec.result.validation, "stats": session.stats()}


@router.get("/integration/export", tags=["results"])
def export(file: str | None = None, session: Session = Depends(deps.get_session)):
    files = session.export_paths()
    if file is None:
        return {"merge_id": session.current_merge().result.merge_id, "files": {k: Path(v).stat().st_size for k, v in files.items()}}
    if file not in files:
        raise NotFoundError(f"Unknown export {file!r}", available=sorted(files))
    media = {"parquet": "application/octet-stream", "csv": "text/csv", "json": "application/json", "txt": "text/plain"}.get(file.rsplit(".", 1)[-1], "application/octet-stream")
    return FileResponse(files[file], filename=file, media_type=media)


# --------------------------------------------------------------------------- chat
@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, request: Request) -> dict[str, Any]:
    session = deps.registry().get_or_create(req.session_id, owner=deps.current_user(request))
    request.state.session_id = session.session_id
    return deps.agent(req.provider).handle(session, req.message)


@router.get("/chat/history", tags=["chat"])
def chat_history(session: Session = Depends(deps.get_session)) -> list[dict[str, Any]]:
    return session.chat_history


def _copy_tree_safe(src: Path, dest: Path) -> None:  # pragma: no cover - utility for demos
    shutil.copytree(src, dest, dirs_exist_ok=True)
