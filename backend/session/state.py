"""Session state and the dataset orchestrator.

A session holds everything a conversation builds up, so the user never has to
repeat themselves:

    Session
     ├── datasets (artifacts) + profiles + value sketches
     ├── schema matches (proposed / approved / rejected) + dataset relationships
     ├── integration graph
     ├── preferences (mode, conflict strategy, threshold, primary key, source priority)
     ├── merge plan
     ├── merges (results, conflicts, lineage, exports) with an undo stack
     └── chat history

Every public method is a deterministic, validated backend operation. The chat
agent (LLM or rule-based) only ever *selects* these operations and their
arguments; it never executes code.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from backend.config import get_settings, load_config
from backend.core.errors import InvalidStateError, NotFoundError, ToolArgumentError
from backend.core.models import ColumnMatch, Conflict, DatasetArtifact, DatasetProfile, DatasetRelationship, MergePlan, MergeResult
from backend.entity_resolution.resolver import FieldSpec, resolve as resolve_entities, review_queue
from backend.graph import algorithms as ga
from backend.graph.explain import explain_column_match, explain_relationship, explain_route
from backend.graph.relationships import infer_relationships
from backend.graph.schema_graph import IntegrationGraph
from backend.graph.serialize import react_flow_view, to_json
from backend.ingestion.loader import ingest_file, ingest_rest
from backend.logging_conf import get_logger
from backend.matching.matcher import MatchingResult, match_schemas
from backend.merge.executor import ExecutionContext, MergeExecutor, relationship_flags
from backend.merge.modes import MODES
from backend.merge.planner import CONFLICT_STRATEGIES, build_plan
from backend.profiling.profiler import profile_dataset
from backend.profiling.sketches import ColumnSketch
from backend.provenance.tracker import column_lineage, prov_document
from backend.storage.duck import parquet_scan, quote_ident
from backend.storage.exports import write_bundle
from backend.storage.workspace import Workspace
from backend.validation.quality_report import build_quality_report
from backend.validation.validators import validate_merge

log = get_logger(__name__)


@dataclass
class MergeRecord:
    result: MergeResult
    plan: MergePlan
    files: dict[str, str]
    conflicts: list[Conflict]
    lineage: dict[str, Any]
    group_stats: dict[str, Any]
    join_stats: list[dict[str, Any]]
    entity_columns: dict[str, dict[str, str]]
    er_pairs: dict[str, list[dict[str, Any]]]
    status: str = "active"  # active | undone | outdated (datasets changed after the merge)
    er_results: dict[str, Any] = field(default_factory=dict)  # group id → ERResult (in memory, for the review queue)
    relationship_flags: list[dict[str, Any]] = field(default_factory=list)  # joins rows depend on (for scenario analysis)


@dataclass
class Preferences:
    mode: str = "balanced"
    conflict_strategy: str = "majority_vote"
    confidence_threshold: float | None = None
    primary_key: str | None = None
    merge_uncertain: bool | None = None
    source_priority: list[str] | None = None

    def overrides(self) -> dict[str, Any]:
        o: dict[str, Any] = {}
        if self.confidence_threshold is not None:
            o["confidence_threshold"] = self.confidence_threshold
        if self.primary_key:
            o["primary_key"] = self.primary_key
        if self.merge_uncertain is not None:
            o["merge_uncertain"] = self.merge_uncertain
        if self.source_priority:
            o["source_priority"] = self.source_priority
        return o


STATE_FILE = "session_state.pkl"
STATE_VERSION = 1


class Session:
    def __init__(self, session_id: str | None = None, workspace_root: str | Path | None = None, config: dict | None = None, owner: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.owner = owner  # user id when API tokens are configured; sessions are private to their owner
        self.created_at = datetime.now(timezone.utc)
        self.workspace = Workspace(workspace_root or get_settings().dfg_workspace, self.session_id)
        self.config = config or load_config()
        self.lock = threading.RLock()
        self.artifacts: dict[str, DatasetArtifact] = {}
        self.profiles: dict[str, DatasetProfile] = {}
        self.sketches: dict[str, dict[str, ColumnSketch]] = {}
        self.matching: MatchingResult | None = None
        self.relationships: list[DatasetRelationship] = []
        self.graph: IntegrationGraph | None = None
        self.discovery_stale = True
        self.plan: MergePlan | None = None
        self.merges: list[MergeRecord] = []
        self.preferences = Preferences(mode=self.config["merge"]["default_mode"], conflict_strategy=self.config["merge"]["default_conflict_strategy"])
        self.decisions: dict[tuple[str, str], str] = {}  # user approvals / rejections of column matches
        self.entity_labels: dict[tuple[tuple[str, int], tuple[str, int]], bool] = {}  # user-labelled record pairs
        self.calibration_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()  # of those, pairs shown as random calibration samples
        self.crosswalks: list[dict[str, Any]] = []  # LLM-proposed, data-verified value crosswalks (matching/crosswalk.py)
        self.crosswalk_proposer = None  # injectable for tests; default: the configured LLM
        self.attribute_checks: list[dict[str, Any]] = []  # row-level verification of attribute correspondences
        self.learned_matcher = None  # session-specific retrained LearnedMatcher (None → shipped model)
        self.chat_history: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    # ================================================================== datasets
    def _event(self, kind: str, **data: Any) -> None:
        self.events.append({"at": datetime.now(timezone.utc).isoformat(), "event": kind, **data})

    def add_file(self, path: str | Path, source_name: str | None = None, options: dict[str, Any] | None = None, profile: bool = True) -> DatasetArtifact:
        with self.lock:
            art = ingest_file(self.workspace, path, source_name, set(self.artifacts), options)
            self.artifacts[art.dataset_id] = art
            if profile:
                self.profile(art.dataset_id)
            self._invalidate()
            self._event("dataset_loaded", dataset_id=art.dataset_id)
            return art

    def add_files(self, path: str | Path, source_name: str | None = None) -> list[DatasetArtifact]:
        """Ingest a file; a SQLite database with several tables becomes one dataset per table."""
        from backend.core.models import SourceType
        from backend.ingestion.detect import detect_format
        from backend.ingestion.readers import list_sqlite_tables

        path = Path(path)
        if detect_format(path) == SourceType.SQLITE:
            tables = list_sqlite_tables(path)
            if len(tables) > 1:
                name = source_name or path.name
                return [self.add_file(path, source_name=f"{name} : {t}", options={"table": t}) for t in tables]
        return [self.add_file(path, source_name=source_name)]

    def add_rest(self, url: str, source_name: str, params: dict[str, Any] | None = None, page_size: int | None = None) -> DatasetArtifact:
        with self.lock:
            art = ingest_rest(self.workspace, url, source_name, set(self.artifacts), params, page_size)
            self.artifacts[art.dataset_id] = art
            self.profile(art.dataset_id)
            self._invalidate()
            return art

    def remove_dataset(self, dataset_id: str) -> None:
        with self.lock:
            self._require_dataset(dataset_id)
            self.artifacts.pop(dataset_id)
            self.profiles.pop(dataset_id, None)
            self.sketches.pop(dataset_id, None)
            self._invalidate()
            self._event("dataset_removed", dataset_id=dataset_id)

    def _invalidate(self) -> None:
        """Datasets changed: discovery, plan and any current merge no longer describe the session's data."""
        self.discovery_stale = True
        self.plan = None
        for rec in self.merges:
            if rec.status == "active":
                rec.status = "outdated"  # outputs stay on disk; results tabs and the agent stop treating it as current
                self._event("merge_outdated", merge_id=rec.result.merge_id)

    def _require_dataset(self, dataset_id: str) -> DatasetArtifact:
        if dataset_id in self.artifacts:
            return self.artifacts[dataset_id]
        for a in self.artifacts.values():
            if dataset_id in (a.source_name, Path(a.source_name).stem):
                return a
        raise NotFoundError(f"Unknown dataset {dataset_id!r}", available=list(self.artifacts))

    def resolve_dataset_id(self, name: str) -> str:
        return self._require_dataset(name).dataset_id

    def list_datasets(self) -> list[dict[str, Any]]:
        return [a.summary() for a in self.artifacts.values()]

    def profile(self, dataset_id: str) -> DatasetProfile:
        with self.lock:
            art = self._require_dataset(dataset_id)
            prof, sk = profile_dataset(art, self.config)
            self.profiles[art.dataset_id] = prof
            self.sketches[art.dataset_id] = sk
            art.profile = prof
            return prof

    def preview(self, dataset_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        art = self._require_dataset(dataset_id)
        limit = max(1, min(int(limit), 500))
        with duckdb.connect() as con:
            cur = con.execute(f"SELECT * FROM {parquet_scan(art.storage_path)} LIMIT {limit} OFFSET {int(offset)}")
            cols = [d[0] for d in cur.description]
            rows = [[_cell(v) for v in r] for r in cur.fetchall()]
        return {"dataset_id": art.dataset_id, "columns": cols, "rows": rows, "total_rows": art.row_count}

    # ================================================================== discovery
    def discover(self, force: bool = False) -> dict[str, Any]:
        with self.lock:
            if len(self.artifacts) < 2:
                raise InvalidStateError("Load at least two datasets before discovering relationships")
            if self.discovery_stale or force or self.matching is None:
                for ds in self.artifacts:
                    if ds not in self.profiles:
                        self.profile(ds)
                started = time.perf_counter()
                accept = self.config["merge_modes"][self.preferences.mode]["min_edge"]
                if self.preferences.confidence_threshold is not None:
                    accept = max(accept, self.preferences.confidence_threshold)
                self.matching = match_schemas(self.profiles, self.sketches, self.config, min_accept=accept, learned=self.learned_matcher, keep_pairs=set(self.decisions),
                                              crosswalks=self._crosswalk_normalizers())
                if self._discover_crosswalks():
                    # second pass: column pairs with verified crosswalks are scored with them
                    self.matching = match_schemas(self.profiles, self.sketches, self.config, min_accept=accept, learned=self.learned_matcher,
                                                  keep_pairs=set(self.decisions), crosswalks=self._crosswalk_normalizers())
                self._apply_decisions()
                self.relationships = infer_relationships(self.profiles, self.matching.matches, self.config)
                from backend.graph.verification import verify_attribute_matches

                self.attribute_checks = verify_attribute_matches(self.relationships, self.artifacts, self.config)
                g = IntegrationGraph(self.config["graph"]["path_cost"])
                for ds, art in self.artifacts.items():
                    g.add_dataset(art, self.profiles[ds])
                g.add_column_matches(self.matching.matches)
                g.add_dataset_relationships(self.relationships)
                self.graph = g
                self.discovery_stale = False
                self.plan = None
                log.info("Graph constructed", **g.stats(), elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
                self._event("discovered", relationships=len(self.relationships))
            return self.discovery_summary()

    def _crosswalk_normalizers(self) -> dict[frozenset[str], str]:
        return {frozenset((cw["left"], cw["right"])): cw["name"] for cw in self.crosswalks if cw.get("accepted")}

    def _discover_crosswalks(self) -> bool:
        """Ask the LLM for value crosswalks between vocabulary columns and keep the ones the data verifies."""
        from backend.matching.crosswalk import llm_proposer, propose_crosswalks

        cfg = self.config["schema_matching"].get("crosswalk", {})
        if not cfg.get("enabled", False) or self.matching is None:
            return False
        proposer = self.crosswalk_proposer
        if proposer is None:
            try:
                from backend.llm.factory import make_provider

                provider = make_provider()
                if provider.is_mock:
                    return False
                proposer = llm_proposer(provider)
            except Exception as exc:  # noqa: BLE001
                log.info("No LLM for crosswalk proposals", error=repr(exc))
                return False
        accepted = {frozenset((m.left.key, m.right.key)) for m in self.matching.matches if m.accepted}
        scores = {frozenset((m.left.key, m.right.key)): m.score for m in self.matching.matches}
        known = {(cw["left"], cw["right"]) for cw in self.crosswalks}
        found = propose_crosswalks(self.profiles, self.sketches, self.config, accepted, scores, proposer,
                                   cache_path=self.workspace.root.parent.parent / "_crosswalk_cache.json")
        new = [cw.to_dict() for cw in found if (cw.left, cw.right) not in known]
        self.crosswalks += new
        if new:
            self._event("crosswalks_checked", proposed=len(new), accepted=sum(1 for c in new if c["accepted"]))
        return any(c["accepted"] for c in new)

    def _apply_decisions(self) -> None:
        if not self.matching:
            return
        for m in self.matching.matches:
            key = tuple(sorted((m.left.key, m.right.key)))
            decision = self.decisions.get(key)  # type: ignore[arg-type]
            if decision == "approved":
                m.accepted, m.status, m.rejection_reason = True, "approved", None
            elif decision == "rejected":
                m.accepted, m.status, m.rejection_reason = False, "rejected", "rejected by the user"

    def _ensure_discovered(self) -> None:
        if self.discovery_stale or self.matching is None or self.graph is None:
            self.discover()

    def discovery_summary(self) -> dict[str, Any]:
        assert self.matching is not None and self.graph is not None
        min_edge = self.config["merge_modes"][self.preferences.mode]["min_edge"]
        accepted = self.matching.accepted()
        return {
            "datasets": [{"dataset_id": a.dataset_id, "source_name": a.source_name, "format": a.source_type.value, "rows": a.row_count, "columns": a.column_count} for a in self.artifacts.values()],
            "relationships": [
                {"left": r.left_dataset, "right": r.right_dataset, "confidence": r.confidence, "band": r.band.value, "join_kind": r.join_kind.value,
                 "keys": [f"{m.left.column} ↔ {m.right.column}" for m in r.key_matches], "explanation": r.explanation}
                for r in self.relationships
            ],
            "mappings": [self._match_summary(m) for m in accepted],
            "rejected_candidates": [self._match_summary(m) for m in self.matching.matches if not m.accepted][:25],
            "components": ga.integration_components(self.graph, min_edge),
            "issues": self.detect_issues(),
            "graph": self.graph.stats(),
            "candidate_pairs": self.matching.candidate_pairs,
            "candidate_strategy": self.matching.candidate_strategy,
            "flooding": self.matching.flooding,
            "elapsed_ms": self.matching.elapsed_ms,
        }

    @staticmethod
    def _match_summary(m: ColumnMatch) -> dict[str, Any]:
        return {
            "left": m.left.key, "right": m.right.key, "confidence": m.score, "band": m.band.value, "relationship": m.relationship,
            "accepted": m.accepted, "status": m.status, "normalizer": m.evidence.normalizer, "rejection_reason": m.rejection_reason,
        }

    def detect_issues(self) -> list[dict[str, Any]]:
        """Data problems a human integrator would want to hear about before merging."""
        issues: list[dict[str, Any]] = []
        for ds, p in self.profiles.items():
            if p.duplicate_rows:
                issues.append({"type": "duplicate_rows", "dataset": ds, "count": p.duplicate_rows, "message": f"{ds}: {p.duplicate_rows} exact duplicate rows"})
            for c in p.columns:
                if c.semantic_type.value in ("DATE", "DATETIME") and c.data_type.value == "string":
                    issues.append({"type": "date_format_inconsistency", "dataset": ds, "column": c.name, "message": f"{ds}.{c.name}: dates stored as text in mixed formats (dominant {c.date_format}, {'day-first' if c.date_dayfirst else 'month-first'})"})
                if c.semantic_type.value == "CURRENCY" and c.data_type.value == "string":
                    issues.append({"type": "currency_representation", "dataset": ds, "column": c.name, "message": f"{ds}.{c.name}: amounts stored as text with currency symbols/codes"})
                if c.null_pct >= 0.2:
                    issues.append({"type": "missing_values", "dataset": ds, "column": c.name, "message": f"{ds}.{c.name}: {c.null_pct:.0%} missing"})
        if self.matching:
            for r in self.relationships:
                if r.join_kind.value == "entity_resolution":
                    issues.append({"type": "no_shared_identifier", "datasets": [r.left_dataset, r.right_dataset], "message": f"{r.left_dataset} and {r.right_dataset} share no reliable identifier; records must be linked probabilistically"})
                for m in r.key_matches:
                    cov = r.evidence.get("coverage")
                    if cov is not None and cov < 0.999 and r.join_kind.value == "lookup":
                        issues.append({"type": "orphan_references", "datasets": [r.left_dataset, r.right_dataset], "message": f"{r.evidence['fact']}: {1 - cov:.1%} of referenced keys do not exist in {r.evidence['dimension']}"})
                        break
        return issues

    # ================================================================== comparisons / explanations
    def compare_schemas(self, left: str, right: str) -> dict[str, Any]:
        self._ensure_discovered()
        a, b = self.resolve_dataset_id(left), self.resolve_dataset_id(right)
        ms = [m for m in self.matching.matches if {m.left.dataset_id, m.right.dataset_id} == {a, b}]  # type: ignore[union-attr]
        weights = self.config["schema_matching"]["weights"]
        return {"left": a, "right": b, "matches": [explain_column_match(m, weights) for m in ms]}

    def find_match(self, left: str, right: str) -> ColumnMatch:
        self._ensure_discovered()

        def parse(ref: str) -> tuple[str | None, str]:
            ref = ref.replace("::", ".")
            if "." in ref:
                ds, col = ref.rsplit(".", 1)
                try:
                    return self.resolve_dataset_id(ds), col
                except NotFoundError:
                    pass
            return None, ref

        (da, ca), (db, cb) = parse(left), parse(right)
        best = None
        for m in self.matching.matches:  # type: ignore[union-attr]
            for (l, r) in ((m.left, m.right), (m.right, m.left)):
                if l.column.lower() == ca.lower() and r.column.lower() == cb.lower() and (da in (None, l.dataset_id)) and (db in (None, r.dataset_id)):
                    if best is None or m.score > best.score:
                        best = m
        if best is None:
            raise NotFoundError(f"No scored correspondence between {left!r} and {right!r} (it may have been pruned below the candidate threshold)")
        return best

    def explain_match(self, left: str, right: str) -> dict[str, Any]:
        return explain_column_match(self.find_match(left, right), self.config["schema_matching"]["weights"])

    def explain_relationship(self, left: str, right: str) -> dict[str, Any]:
        self._ensure_discovered()
        a, b = self.resolve_dataset_id(left), self.resolve_dataset_id(right)
        for r in self.relationships:
            if {r.left_dataset, r.right_dataset} == {a, b}:
                return explain_relationship(r, self.config["schema_matching"]["weights"])
        route = ga.best_route(self.graph, a, b)  # type: ignore[arg-type]
        weights = self.config["schema_matching"]["weights"]
        candidates = sorted((m for m in self.matching.matches if {m.left.dataset_id, m.right.dataset_id} == {a, b}), key=lambda m: -m.score)[:5]  # type: ignore[union-attr]
        lines = [f"{a} and {b} are not directly linked under the current settings (mode {self.preferences.mode}"
                 + (f", minimum confidence {self.preferences.confidence_threshold:.0%}" if self.preferences.confidence_threshold else "") + ")."]
        if candidates:
            lines.append("Their strongest column correspondences were:")
            lines += [f"- {m.left.column} ↔ {m.right.column}: {m.score:.2f} — {'accepted' if m.accepted else m.rejection_reason}" for m in candidates]
        if route.get("found"):
            lines.append(explain_route(route))
        return {"datasets": [a, b], "direct_relationship": None, "route": route, "candidates": [explain_column_match(m, weights) for m in candidates],
                "summary": "\n".join(lines)}

    def route(self, source: str, target: str, cost_mode: str | None = None) -> dict[str, Any]:
        self._ensure_discovered()
        a, b = self.resolve_dataset_id(source), self.resolve_dataset_id(target)
        r = ga.best_route(self.graph, a, b, cost_mode=cost_mode)  # type: ignore[arg-type]
        r["explanation"] = explain_route(r)
        return r

    def decide_match(self, left: str, right: str, decision: str) -> dict[str, Any]:
        if decision not in ("approved", "rejected"):
            raise ToolArgumentError("decision must be 'approved' or 'rejected'")
        with self.lock:
            m = self.find_match(left, right)
            self.decisions[tuple(sorted((m.left.key, m.right.key)))] = decision  # type: ignore[index]
            self._apply_decisions()
            self.relationships = infer_relationships(self.profiles, self.matching.matches, self.config)  # type: ignore[union-attr]
            g = IntegrationGraph(self.config["graph"]["path_cost"])
            for ds, art in self.artifacts.items():
                g.add_dataset(art, self.profiles[ds])
            g.add_column_matches(self.matching.matches)  # type: ignore[union-attr]
            g.add_dataset_relationships(self.relationships)
            self.graph = g
            self.plan = None
            return self._match_summary(m)

    def retrain_matcher(self, label_weight: float = 10.0) -> dict[str, Any]:
        """Refit the learned column matcher on its base training set plus this session's approve/reject decisions."""
        import numpy as np

        from backend.config import PROJECT_ROOT
        from backend.matching.learned import load_default, pair_features

        with self.lock:
            self._ensure_discovered()
            if not self.decisions:
                raise InvalidStateError("No mapping decisions yet: approve or reject some mappings first")
            base = self.learned_matcher or load_default(PROJECT_ROOT / self.config["schema_matching"].get("learned_model", "models/column_matcher.joblib"))
            if base is None:
                raise InvalidStateError("No trained matcher found; run experiments/train_matcher.py")
            X, y, used = [], [], []
            for (ka, kb), decision in self.decisions.items():
                ev = self.matching.evidence.get((ka, kb)) or self.matching.evidence.get((kb, ka))  # type: ignore[union-attr]
                if ev is None:
                    continue
                cols = self.matching.columns  # type: ignore[union-attr]
                X.append(pair_features(ev, cols[ka], cols[kb]))
                y.append(1 if decision == "approved" else 0)
                used.append(f"{ka} ↔ {kb}: {decision}")
            if not X:
                raise InvalidStateError("None of the decided mappings are among the scored candidate pairs")
            self.learned_matcher = base.retrain_with_decisions(np.array(X, dtype=float), np.array(y), weight=label_weight)
            self.learned_matcher.save(self.workspace.root / "column_matcher.joblib", training_data=getattr(base, "_training_data", None))
            self.discovery_stale = True
            scorer = self.config["schema_matching"].get("scorer", "weighted")
            self._event("matcher_retrained", labels=len(y))
            return {
                "labels_used": used,
                "metadata": self.learned_matcher.metadata,
                "active_scorer": scorer,
                "note": None if scorer != "weighted" else "The retrained model is used when the scorer preference is 'learned' or 'blend' (currently 'weighted').",
            }

    def graph_view(self, level: str = "dataset") -> dict[str, Any]:
        self._ensure_discovered()
        if level == "schema":
            from backend.graph.schema_map import schema_map_view

            return schema_map_view(self.artifacts, self.profiles, self.matching.matches, self.relationships, self.config, self.plan)  # type: ignore[union-attr]
        tree = set()
        if self.plan:
            tree = {frozenset((e["parent"], e["child"])) for e in self.plan.integration_tree}
        view = react_flow_view(self.graph, level, tree)  # type: ignore[arg-type]
        view["stats"] = self.graph.stats()  # type: ignore[union-attr]
        return view

    def entity_match_preview(self, left: str, right: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Run entity resolution between two datasets (or dedup one) without merging."""
        self._ensure_discovered()
        a = self.resolve_dataset_id(left)
        b = self.resolve_dataset_id(right) if right else None
        datasets = [a] + ([b] if b and b != a else [])
        from backend.graph.algorithms import column_groups
        from backend.core.models import SemanticType

        fields = []
        if b:
            for g in column_groups(self.matching.matches, set(datasets)):  # type: ignore[union-attr]
                cols = {d: c for d, c in g}
                if len(cols) < 2:
                    continue
                st = self.profiles[a].column(cols[a]).semantic_type
                if st.value in ("CURRENCY", "NUMERIC"):
                    continue
                best = max((m for m in self.matching.matches if m.accepted and (m.left.dataset_id, m.left.column) in g and (m.right.dataset_id, m.right.column) in g), key=lambda m: m.score)  # type: ignore[union-attr]
                fields.append(FieldSpec(cols[a], st, cols, best.evidence.normalizer))
        else:
            for c in self.profiles[a].columns:
                if c.semantic_type in (SemanticType.NAME, SemanticType.EMAIL, SemanticType.PHONE, SemanticType.CITY):
                    fields.append(FieldSpec(c.name, c.semantic_type, {a: c.name}, "basic"))
        if not fields:
            raise InvalidStateError(f"No comparable attributes between {datasets}")
        t = self.config["merge_modes"][self.preferences.mode]
        er = resolve_entities({d: self.artifacts[d].storage_path for d in datasets}, fields, self.config,
                              auto_threshold=t.get("entity_merge", t["auto_merge"]), review_threshold=t.get("entity_review", t["review"]))
        dup_clusters = [c for c in er.clusters if c.size > 1]
        return {
            "datasets": datasets,
            "fields": [{"name": f.name, "semantic_type": f.semantic_type.value, "columns": f.columns, "normalizer": f.normalizer} for f in fields],
            "stats": er.stats,
            "model": er.model.to_dict(),
            "top_pairs": [p.model_dump() for p in er.pairs[:limit]],
            "uncertain_pairs": [p.model_dump() for p in er.pairs if t.get("entity_review", t["review"]) <= p.probability < t.get("entity_merge", t["auto_merge"])][:limit],
            "clusters_with_multiple_records": len(dup_clusters),
            "sample_clusters": [c.model_dump() for c in dup_clusters[:limit]],
        }

    # ================================================================== preferences / planning
    def set_preferences(self, **kw: Any) -> dict[str, Any]:
        with self.lock:
            p = self.preferences
            if kw.get("scorer") is not None:
                if kw["scorer"] not in ("weighted", "learned", "blend"):
                    raise ToolArgumentError("scorer must be one of ('weighted', 'learned', 'blend')")
                self.config["schema_matching"]["scorer"] = kw["scorer"]
            if kw.get("mode") is not None:
                if kw["mode"] not in MODES:
                    raise ToolArgumentError(f"mode must be one of {MODES}")
                p.mode = kw["mode"]
            if kw.get("conflict_strategy") is not None:
                if kw["conflict_strategy"] not in CONFLICT_STRATEGIES:
                    raise ToolArgumentError(f"conflict_strategy must be one of {CONFLICT_STRATEGIES}")
                p.conflict_strategy = kw["conflict_strategy"]
            if "confidence_threshold" in kw and kw["confidence_threshold"] is not None:
                c = float(kw["confidence_threshold"])
                if c > 1:
                    c = c / 100.0
                if not 0 < c <= 1:
                    raise ToolArgumentError("confidence_threshold must be between 0 and 1 (or a percentage)")
                p.confidence_threshold = c
            if kw.get("clear_threshold"):
                p.confidence_threshold = None
            if kw.get("primary_key") is not None:
                pk = kw["primary_key"]
                if pk:
                    hits = [c.name for prof in self.profiles.values() for c in prof.columns if c.name.lower() == pk.split(".")[-1].lower()]
                    if not hits:
                        raise ToolArgumentError(f"Column {pk!r} not found in any dataset")
                p.primary_key = pk or None
            if kw.get("merge_uncertain") is not None:
                p.merge_uncertain = bool(kw["merge_uncertain"])
            if kw.get("source_priority") is not None:
                p.source_priority = [self.resolve_dataset_id(d) for d in kw["source_priority"]]
            self.discovery_stale = True
            self.plan = None
            return self.preferences_dict()

    def preferences_dict(self) -> dict[str, Any]:
        p = self.preferences
        return {"mode": p.mode, "conflict_strategy": p.conflict_strategy, "confidence_threshold": p.confidence_threshold, "primary_key": p.primary_key,
                "merge_uncertain": p.merge_uncertain, "source_priority": p.source_priority, "thresholds": self.config["merge_modes"][p.mode],
                "scorer": self.config["schema_matching"].get("scorer", "weighted")}

    def make_plan(self, mode: str | None = None, conflict_strategy: str | None = None, datasets: list[str] | None = None) -> MergePlan:
        with self.lock:
            if mode or conflict_strategy:
                self.set_preferences(mode=mode, conflict_strategy=conflict_strategy)
            self._ensure_discovered()
            overrides = self.preferences.overrides()
            if datasets:
                overrides["datasets"] = [self.resolve_dataset_id(d) for d in datasets]
            self.plan = build_plan(self.artifacts, self.profiles, self.matching.matches, self.relationships, self.graph, self.config,  # type: ignore[arg-type,union-attr]
                                   mode=self.preferences.mode, conflict_strategy=self.preferences.conflict_strategy, overrides=overrides)
            self._event("planned", plan_id=self.plan.plan_id)
            return self.plan

    # ================================================================== execution
    def execute(self, plan_id: str | None = None, write_csv: bool = True) -> MergeRecord:
        with self.lock:
            if self.plan is None or (plan_id and self.plan.plan_id != plan_id):
                if plan_id:
                    raise InvalidStateError(f"Plan {plan_id} is no longer current; generate a new plan")
                self.make_plan()
            plan = self.plan
            assert plan is not None and self.matching is not None and self.graph is not None
            started = time.perf_counter()
            con = duckdb.connect(":memory:")
            try:
                engine = self.config.setdefault("engine", {})
                if not engine.get("temp_directory"):
                    engine["temp_directory"] = str(self.workspace.root / "_duckdb_tmp")
                executor = MergeExecutor(plan, self.artifacts, self.profiles, self.config, con)
                flags: list[dict[str, Any]] = []
                executor.entity_labels = dict(self.entity_labels)
                executor.calibration_pairs = set(self.calibration_pairs)
                ctx = executor.run()
                t0 = time.perf_counter()
                executor.materialize_output(self.workspace.merge_dir(ctx.merge_id) / "unified_dataset.parquet")
                ctx.timings_ms["materialize"] = round((time.perf_counter() - t0) * 1000, 1)
                log.info("Merge completed", merge_id=ctx.merge_id, conflicts=len(ctx.conflicts))
                rows = con.execute("SELECT count(*) FROM unified").fetchone()[0]
                columns = [r[0] for r in con.execute("DESCRIBE unified").fetchall()]
                t0 = time.perf_counter()
                validation = validate_merge(con, ctx, plan, self.artifacts)
                ctx.timings_ms["validation"] = round((time.perf_counter() - t0) * 1000, 1)
                tau = float(plan.thresholds.get("auto_merge", 0.9))
                exp_rows, below = con.execute(f"SELECT sum(_match_probability), count(*) FILTER (WHERE _match_probability < {tau}) FROM unified").fetchone()
                quality = build_quality_report(plan, self.artifacts, self.profiles, self.matching.matches, ctx, rows, validation,
                                               probability={"expected_correct_rows": round(float(exp_rows or 0.0), 2), "rows_below_threshold": int(below or 0)})
                lineage = column_lineage(ctx.spec, plan)
                flags = relationship_flags(ctx.spec)
                prov = prov_document(plan, self.artifacts, ctx.merge_id, self.session_id, rows, ctx.group_stats, ctx.timings_ms)
                entity_columns = {gid: {c.view_name: c.output for c in self._group_spec(ctx, gid).columns} for gid in ctx.entity_tables}
                matched_rows = quality["rows_matched_in_all_lookups"] if quality["rows_matched_in_all_lookups"] is not None else rows
                result = MergeResult(
                    merge_id=ctx.merge_id, plan_id=plan.plan_id, output_path="", row_count=rows, column_count=len(columns), columns=columns,
                    matched_rows=matched_rows, unmatched_rows=rows - matched_rows, duplicates_resolved=quality["duplicate_entities_resolved"],
                    conflicts=len(ctx.conflicts), entity_clusters=quality["resolved_entities"], elapsed_ms=0.0,
                    stage_timings_ms=ctx.timings_ms, validation=validation, quality_report=quality,
                )
                merge_report = {
                    "merge_id": ctx.merge_id, "session_id": self.session_id, "plan": plan.model_dump(mode="json"), "result": result.model_dump(mode="json"),
                    "join_stats": ctx.join_stats, "entity_groups": ctx.group_stats, "source_accuracy": ctx.source_accuracy, "warnings": plan.warnings + ctx.warnings,
                    "entity_table_columns": entity_columns, "preferences": self.preferences_dict(),
                    "datasets": {d: {"source": a.source_name, "format": a.source_type.value, "rows": a.row_count, "sha256": a.checksum} for d, a in self.artifacts.items() if d in plan.datasets},
                }
                self.graph.add_derived(ctx.merge_id, "unified dataset", plan.datasets, rows)
                out_dir = self.workspace.merge_dir(ctx.merge_id)
                t0 = time.perf_counter()
                recorded_at = result.created_at.replace(tzinfo=None)
                for hist in ctx.histories.values():
                    for h in hist:
                        h["recorded_at"] = recorded_at
                files = write_bundle(
                    out_dir, con, ctx.entity_tables, merge_report, {"column_lineage": lineage, "prov": prov, "row_lineage_columns": [c for c in columns if c.startswith("_src_") or c == "_record_id"]},
                    ctx.conflicts, ctx.cell_provenance, self.schema_mapping_export(), to_json(self.graph), quality["text"], write_csv=write_csv, histories=ctx.histories,
                )
                ctx.timings_ms["export"] = round((time.perf_counter() - t0) * 1000, 1)
            finally:
                con.close()
            result.output_path = files["unified_dataset.parquet"]
            result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            er_pairs = {gid: [p.model_dump() for p in er.pairs[:2000]] for gid, er in ctx.er_results.items()}
            record = MergeRecord(result, plan, files, ctx.conflicts, lineage, ctx.group_stats, ctx.join_stats, entity_columns, er_pairs, er_results=dict(ctx.er_results),
                                 relationship_flags=flags)
            self.merges.append(record)
            self._event("merged", merge_id=ctx.merge_id, rows=rows)
            log.info("Merge exported", merge_id=ctx.merge_id, files=len(files), validation_passed=validation["passed"], elapsed_ms=result.elapsed_ms)
            return record

    @staticmethod
    def _group_spec(ctx: ExecutionContext, gid: str):
        stack = [ctx.spec]
        while stack:
            node = stack.pop()
            if node.group is not None and node.group.group_id == gid:
                return node.group
            stack += [a.child for a in node.attachments]
            if node.group is not None:
                stack += list(node.group.members.values())
        raise KeyError(gid)

    def aggregate_with_uncertainty(self, measure: str | None = None, group_by: str | None = None, agg: str = "sum", limit: int = 50) -> dict[str, Any]:
        """Aggregate the unified dataset with linkage uncertainty propagated (expected value, 95% interval, certain-only)."""
        from backend.provenance.uncertainty import aggregate_with_uncertainty

        rec = self.current_merge()
        tau = float(rec.plan.thresholds.get("entity_merge", rec.plan.thresholds.get("auto_merge", 0.9)))
        scenarios = self._relationship_scenarios(rec, [c for c in (measure, group_by) if c])
        out = aggregate_with_uncertainty(Path(rec.files["unified_dataset.parquet"]).as_posix(), measure, group_by, agg, tau, limit, scenarios)
        out["merge_id"] = rec.result.merge_id
        return out

    def _relationship_scenarios(self, rec: MergeRecord, columns: list[str]) -> list[dict[str, Any]]:
        """Uncertain relationships (confidence below HIGH) that the given output columns are derived through."""
        high = float(self.config["thresholds"]["high"])
        out = []
        for f in getattr(rec, "relationship_flags", []):
            if f["confidence"] >= high:
                continue
            depends = False
            chain = [tuple(x) for x in f.get("chain", [[f["child"], f["role"]]])]
            for col in columns:
                joins = [(h.get("dataset"), h.get("role")) for h in (rec.lineage.get(col) or {}).get("path", []) if h.get("operation") in ("lookup", "aggregate")]
                if joins[: len(chain)] == chain:  # the column is derived through exactly this chain of joins
                    depends = True
            if depends:
                label = f"{f['parent']} ← {f['child']}" + (f" ({f['role']})" if f["role"] else "")
                out.append({"relationship": label, "flag_column": f["flag_column"], "src_row_column": f.get("src_row_column"), "confidence": f["confidence"]})
        return out

    def entity_history(self, entity_id: str | None = None, attribute: str | None = None, as_of: str | None = None, limit: int = 200,
                       as_known_at: str | None = None) -> dict[str, Any]:
        """Bitemporal attribute history of fused entities.

        ``as_of`` (valid time): the values that were true at that time.
        ``as_known_at`` (transaction time): what the integration knew then, i.e. the history written by the latest
        merge executed at or before that time, including merges that were later undone or outdated."""
        rec = self.current_merge() if as_known_at is None else self._merge_known_at(as_known_at)
        files = sorted(k for k in rec.files if k.startswith("entity_history_"))
        if not files:
            return {"merge_id": rec.result.merge_id, "rows": [], "total": 0, "note": "No timestamped records were fused, so there is no attribute history."}
        where, params = [], []
        if entity_id:
            where.append("entity_id = ?")
            params.append(entity_id)
        if attribute:
            where.append("attribute = ?")
            params.append(attribute)
        if as_of:
            try:
                ts = datetime.fromisoformat(as_of)
            except ValueError as e:
                raise ToolArgumentError(f"as_of must be an ISO date or datetime, got {as_of!r}") from e
            where.append("valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)")
            params += [ts, ts]
        scan = " UNION ALL ".join(f"SELECT * FROM read_parquet('{Path(rec.files[f]).as_posix()}')" for f in files)
        sql = f"SELECT * FROM ({scan}) {'WHERE ' + ' AND '.join(where) if where else ''}"
        con = duckdb.connect(":memory:")
        try:
            total = con.execute(f"SELECT count(*) FROM ({sql})", params).fetchone()[0]
            multi = con.execute(f"SELECT count(*) FROM (SELECT entity_id, attribute FROM ({sql}) WHERE valid_from IS NOT NULL GROUP BY 1, 2 HAVING count(*) > 1)", params).fetchone()[0]
            cur = con.execute(f"SELECT * FROM ({sql}) ORDER BY entity_id, attribute, valid_from NULLS LAST LIMIT {int(limit)}", params)
            names = [d[0] for d in cur.description]
            rows = [{k: _cell(v) for k, v in zip(names, r)} for r in cur.fetchall()]
        finally:
            con.close()
        return {"merge_id": rec.result.merge_id, "recorded_at": rec.result.created_at.isoformat(), "total": total, "attributes_with_changes": multi,
                "as_of": as_of, "as_known_at": as_known_at, "rows": rows}

    def _merge_known_at(self, as_known_at: str) -> MergeRecord:
        try:
            t = datetime.fromisoformat(as_known_at)
        except ValueError as e:
            raise ToolArgumentError(f"as_known_at must be an ISO date or datetime, got {as_known_at!r}") from e
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        known = [r for r in self.merges if r.result.created_at <= t and any(Path(p).exists() for k, p in r.files.items() if k.startswith("entity_history_"))]
        if not known:
            raise InvalidStateError(f"No merge with attribute history had been executed by {as_known_at}")
        return max(known, key=lambda r: r.result.created_at)

    # ================================================================== active learning (entity links)
    def _record_values(self, dataset_id: str, row: int, columns: list[str | None]) -> dict[str, Any]:
        art = self._require_dataset(dataset_id)
        cols = list(dict.fromkeys(c for c in columns if c))
        if not cols:
            return {}
        con = duckdb.connect(":memory:")
        try:
            r = con.execute(f"SELECT {', '.join(quote_ident(c) for c in cols)} FROM read_parquet('{Path(art.storage_path).as_posix()}', file_row_number=true) "
                            "WHERE file_row_number = ?", [int(row)]).fetchone()
        finally:
            con.close()
        return {c: _cell(v) for c, v in zip(cols, r)} if r else {}

    def entity_review_queue(self, limit: int = 20, random_share: float = 0.3, seed: int = 0) -> dict[str, Any]:
        """Record pairs to label next.

        Most pairs come from margin sampling around the auto-merge threshold (they change clustering
        decisions); a ``random_share`` of the queue is drawn uniformly at random. Only the random ones
        recalibrate match probabilities, because uncertainty-sampled labels are a biased sample.
        """
        import random as _random

        rec = self.current_merge()
        n_random = int(round(limit * random_share))
        rng = _random.Random(seed + len(self.entity_labels))
        items: list[dict[str, Any]] = []
        for gid, er in rec.er_results.items():
            group = next((g for g in rec.plan.entity_groups if g["group_id"] == gid), None)
            if group is None:
                continue
            fields = {f["name"]: f["columns"] for f in group["fields"]}
            pairs = {((p.left_dataset, p.left_row), (p.right_dataset, p.right_row)): p for p in er.pairs}
            labelled = set(self.entity_labels)
            chosen = review_queue(er, limit=limit - n_random, threshold=group["auto_threshold"], exclude=labelled)
            for q in chosen:
                q["sampling"] = "uncertainty"
            taken = {((q["left"]["dataset"], q["left"]["row"]), (q["right"]["dataset"], q["right"]["row"])) for q in chosen}
            pool = sorted(k for k in er.pair_probabilities if k not in labelled and k not in taken and (k[1], k[0]) not in labelled)
            for a, b in rng.sample(pool, min(n_random, len(pool))):
                p = er.pair_probabilities[(a, b)]
                same = er.assignment.get(a) is not None and er.assignment.get(a) == er.assignment.get(b)
                chosen.append({"left": {"dataset": a[0], "row": a[1]}, "right": {"dataset": b[0], "row": b[1]}, "probability": round(p, 4),
                               "same_entity_now": same, "score": 0.0, "sampling": "random"})
            for q in chosen:
                a, b = (q["left"]["dataset"], q["left"]["row"]), (q["right"]["dataset"], q["right"]["row"])
                pair = pairs.get((a, b)) or pairs.get((b, a))
                for side, (ds, row) in (("left", a), ("right", b)):
                    q[side]["source"] = self.artifacts[ds].source_name if ds in self.artifacts else ds
                    raw = self._record_values(ds, row, [cols.get(ds) for cols in fields.values()])
                    q[side]["values"] = {name: raw.get(cols[ds]) if ds in cols else None for name, cols in fields.items()}
                q["group_id"] = gid
                q["field_levels"] = pair.field_levels if pair else {}
                items.append(q)
        items.sort(key=lambda x: (x["sampling"] != "uncertainty", -x["score"]))
        return {"merge_id": rec.result.merge_id, "labels": len(self.entity_labels), "calibration_labels": len(self.calibration_pairs), "pairs": items[:limit],
                "note": "Labels become must-/cannot-link constraints on the next merge; randomly sampled pairs also recalibrate match probabilities."}

    def label_entity_link(self, left: dict[str, Any], right: dict[str, Any], decision: str, sampling: str = "uncertainty") -> dict[str, Any]:
        if decision not in ("match", "non_match"):
            raise ToolArgumentError("decision must be 'match' or 'non_match'")
        with self.lock:
            a = (self.resolve_dataset_id(str(left["dataset"])), int(left["row"]))
            b = (self.resolve_dataset_id(str(right["dataset"])), int(right["row"]))
            for ds, row in (a, b):
                if not 0 <= row < self.artifacts[ds].row_count:
                    raise ToolArgumentError(f"Row {row} is out of range for {self.artifacts[ds].source_name}")
            if a == b:
                raise ToolArgumentError("A record cannot be labelled against itself")
            key = (a, b) if a < b else (b, a)
            self.entity_labels[key] = decision == "match"
            if sampling == "random":
                self.calibration_pairs.add(key)
            else:
                self.calibration_pairs.discard(key)
            self.plan = None  # the next merge re-plans and re-resolves with the labels
            self._event("entity_label", left=list(a), right=list(b), decision=decision, sampling=sampling)
            return {"left": {"dataset": a[0], "row": a[1]}, "right": {"dataset": b[0], "row": b[1]}, "decision": decision, "sampling": sampling,
                    "labels": len(self.entity_labels), "note": "Applied on the next merge as a must-link / cannot-link constraint."}

    def current_merge(self) -> MergeRecord:
        for rec in reversed(self.merges):
            if rec.status == "active":
                return rec
        raise InvalidStateError("No merge has been executed yet")

    def undo(self) -> dict[str, Any]:
        with self.lock:
            rec = self.current_merge()
            rec.status = "undone"
            src = Path(rec.files["unified_dataset.parquet"]).parent
            dest = self.workspace.merges / "_undone" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            rec.files = {k: str(dest / Path(v).name) for k, v in rec.files.items()}
            self._event("undone", merge_id=rec.result.merge_id)
            previous = next((r for r in reversed(self.merges) if r.status == "active"), None)
            return {"undone": rec.result.merge_id, "archived_to": str(dest), "current_merge": previous.result.merge_id if previous else None,
                    "note": "Source datasets were never modified; the undone outputs are archived, not deleted."}

    # ================================================================== results
    def output_preview(self, limit: int = 50, offset: int = 0, columns: list[str] | None = None) -> dict[str, Any]:
        rec = self.current_merge()
        limit = max(1, min(int(limit), 1000))
        with duckdb.connect() as con:
            available = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {parquet_scan(rec.result.output_path)}").fetchall()]
            sel = [c for c in (columns or available) if c in available] or available
            cur = con.execute(f"SELECT {', '.join(quote_ident(c) for c in sel)} FROM {parquet_scan(rec.result.output_path)} LIMIT {limit} OFFSET {int(offset)}")
            rows = [[_cell(v) for v in r] for r in cur.fetchall()]
        return {"merge_id": rec.result.merge_id, "columns": sel, "rows": rows, "total_rows": rec.result.row_count}

    def conflicts(self, limit: int = 50, attribute: str | None = None, entity_id: str | None = None) -> dict[str, Any]:
        rec = self.current_merge()
        items = [c for c in rec.conflicts if (attribute is None or c.attribute == attribute) and (entity_id is None or c.entity_id == entity_id)]
        by_attr: dict[str, int] = {}
        for c in rec.conflicts:
            by_attr[c.attribute] = by_attr.get(c.attribute, 0) + 1
        return {"total": len(rec.conflicts), "matching": len(items), "by_attribute": by_attr, "strategy": rec.plan.conflict_strategy,
                "conflicts": [c.model_dump(mode="json") for c in items[: max(1, min(limit, 1000))]]}

    def duplicates(self, limit: int = 20) -> dict[str, Any]:
        rec = self.current_merge()
        out = []
        for gid, pairs in rec.er_pairs.items():
            st = rec.group_stats[gid]
            within = [p for p in pairs if p["left_dataset"] == p["right_dataset"] and p["probability"] >= rec.plan.thresholds.get("entity_merge", rec.plan.thresholds["auto_merge"])]
            out.append({"group": gid, "datasets": st["datasets"], "within_dataset_duplicates_resolved": st["within_dataset_duplicates_resolved"],
                        "cross_source_links": st["cross_source_links"], "examples": within[:limit]})
        exact = {d: self.profiles[d].duplicate_rows for d in rec.plan.datasets if self.profiles[d].duplicate_rows}
        return {"exact_duplicate_rows": exact, "entity_groups": out}

    def provenance(self, column: str | None = None) -> dict[str, Any]:
        rec = self.current_merge()
        if column:
            if column not in rec.lineage:
                raise NotFoundError(f"Column {column!r} is not in the unified dataset", available=list(rec.lineage))
            return {"column": column, **rec.lineage[column]}
        return {"merge_id": rec.result.merge_id, "columns": rec.lineage, "row_lineage_columns": [c for c in rec.result.columns if c.startswith("_src_") or c == "_record_id"],
                "files": {k: v for k, v in rec.files.items() if k in ("provenance.json", "cell_provenance.parquet")}}

    def lineage_report(self) -> str:
        rec = self.current_merge()
        lines = [f"DATA LINEAGE REPORT — merge {rec.result.merge_id}", f"Grain: {rec.plan.grain}", ""]
        for col, info in rec.lineage.items():
            if col.startswith("_"):
                continue
            srcs = ", ".join(f"{s['dataset']}.{s['column']}" + (f" [{s['transformation']}]" if s.get("transformation") else "") for s in info["sources"])
            ops = " → ".join(h["operation"] + (f"({h['role']})" if h.get("role") else "") for h in info["path"])
            lines.append(f"unified.{col}\n    ← {srcs}\n    via {ops}")
        return "\n".join(lines)

    def schema_mapping_export(self) -> dict[str, Any]:
        weights = self.config["schema_matching"]["weights"]
        return {
            "weights": weights,
            "thresholds": self.config["thresholds"],
            "candidate_strategy": self.matching.candidate_strategy if self.matching else None,
            "flooding": self.matching.flooding if self.matching else None,
            "mappings": [explain_column_match(m, weights) for m in (self.matching.matches if self.matching else [])],
            "dataset_relationships": [r.model_dump(mode="json", exclude={"key_matches", "attribute_matches"}) | {"keys": [f"{m.left.key} ↔ {m.right.key}" for m in r.key_matches]} for r in self.relationships],
            "canonical_schema": [c.model_dump(mode="json") for c in (self.plan.canonical_schema if self.plan else [])],
        }

    def export_paths(self) -> dict[str, str]:
        return self.current_merge().files

    def stats(self) -> dict[str, Any]:
        rec = self.current_merge()
        q = rec.result.quality_report
        return {
            "merge_id": rec.result.merge_id, "rows": rec.result.row_count, "matched_rows": rec.result.matched_rows, "unmatched_rows": rec.result.unmatched_rows,
            "match_percentage": round(100 * rec.result.matched_rows / rec.result.row_count, 2) if rec.result.row_count else 0.0,
            "joins": q["joins"], "integration_coverage": q["integration_coverage"], "conflicts": rec.result.conflicts,
            "entities": q["resolved_entities"], "linked_entity_records": q["linked_entity_records"], "entity_records": q["entity_records"],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "datasets": self.list_datasets(),
            "discovered": not self.discovery_stale and self.matching is not None,
            "preferences": self.preferences_dict(),
            "plan_id": self.plan.plan_id if self.plan else None,
            "merges": [{"merge_id": r.result.merge_id, "status": r.status, "rows": r.result.row_count, "conflicts": r.result.conflicts, "created_at": r.result.created_at.isoformat()} for r in self.merges],
            "chat_turns": len(self.chat_history),
            "owner": self.owner,
        }

    # ================================================================== persistence
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("lock", None)  # locks are process-local
        state["crosswalk_proposer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.RLock()
        self.__dict__.setdefault("owner", None)
        self.__dict__.setdefault("crosswalks", [])
        self.__dict__.setdefault("crosswalk_proposer", None)
        from backend.matching.crosswalk import Crosswalk, register

        for cw in self.crosswalks:  # runtime normalisers are process-local: re-register after a restart
            if cw.get("accepted"):
                register(Crosswalk(**cw))

    def persist(self) -> None:
        """Write the whole session (datasets, profiles, discovery, labels, decisions, merges, chat) atomically.

        The pickle lives inside the session's own workspace. It is only ever read back from the
        configured workspace directory, which must not be writable by untrusted users."""
        import os
        import pickle

        with self.lock:
            path = self.workspace.root / STATE_FILE
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump({"version": STATE_VERSION, "session": self}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            self.save_snapshot()

    @classmethod
    def restore(cls, workspace_root: str | Path, session_id: str) -> "Session | None":
        import pickle

        path = Path(workspace_root) / "sessions" / session_id / STATE_FILE
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                blob = pickle.load(fh)  # noqa: S301 - written by this application into its own workspace
        except Exception as exc:  # noqa: BLE001 - a corrupt or incompatible state file must not break the API
            log.warning("Could not restore session; starting fresh", session_id=session_id, error=repr(exc))
            return None
        if not isinstance(blob, dict) or blob.get("version") != STATE_VERSION:
            log.warning("Session state has an incompatible version; starting fresh", session_id=session_id)
            return None
        session: Session = blob["session"]
        session.workspace = Workspace(workspace_root, session_id)
        log.info("Session restored from disk", session_id=session_id, datasets=len(session.artifacts), merges=len(session.merges))
        return session

    def save_snapshot(self) -> None:
        (self.workspace.root / "session.json").write_text(json.dumps(self.snapshot(), indent=2, default=str), encoding="utf-8")


def _cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, str, bool)):
        return v
    if isinstance(v, float):
        return None if v != v else v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


class SessionRegistry:
    """In-memory sessions backed by the workspace: a session not in memory is restored from disk on first use.

    ``owner``: with API tokens configured, each session belongs to the user who created it; other users
    get "not found" (the session's existence is not revealed)."""

    _ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = Path(workspace_root or get_settings().dfg_workspace)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _load(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s is None and self._ID.match(session_id):
            s = Session.restore(self.workspace_root, session_id)
            if s is not None:
                self._sessions[session_id] = s
        return s

    @staticmethod
    def _visible(s: Session, owner: str | None) -> bool:
        return owner is None or s.owner is None or s.owner == owner

    def create(self, owner: str | None = None) -> Session:
        with self._lock:
            s = Session(workspace_root=self.workspace_root, owner=owner)
            self._sessions[s.session_id] = s
        s.persist()
        return s

    def get(self, session_id: str, owner: str | None = None) -> Session:
        with self._lock:
            s = self._load(session_id)
        if s is None or not self._visible(s, owner):
            raise NotFoundError(f"Unknown session {session_id!r}")
        return s

    def get_or_create(self, session_id: str | None, owner: str | None = None) -> Session:
        if session_id and not self._ID.match(session_id):
            raise ToolArgumentError("session id may contain only letters, digits, '_' and '-' (max 64)")
        with self._lock:
            s = self._load(session_id) if session_id else None
            if s is not None:
                if not self._visible(s, owner):
                    raise NotFoundError(f"Unknown session {session_id!r}")
                return s
            s = Session(session_id=session_id, workspace_root=self.workspace_root, owner=owner)
            self._sessions[s.session_id] = s
            return s

    def persist(self, session_id: str) -> None:
        s = self._sessions.get(session_id)
        if s is not None:
            s.persist()

    def delete(self, session_id: str, owner: str | None = None) -> None:
        s = self.get(session_id, owner)
        with self._lock:
            self._sessions.pop(session_id, None)
        s.workspace.remove()

    def list(self, owner: str | None = None) -> list[dict[str, Any]]:
        """Sessions in memory plus those persisted on disk (read from their snapshot, without loading them)."""
        out = {sid: s.snapshot() for sid, s in self._sessions.items() if self._visible(s, owner)}
        base = self.workspace_root / "sessions"
        if base.exists():
            for d in base.iterdir():
                snap = d / "session.json"
                if d.name in out or not snap.exists() or not (d / STATE_FILE).exists():
                    continue
                try:
                    data = json.loads(snap.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                if owner is None or data.get("owner") in (None, owner):
                    out[d.name] = data
        return list(out.values())
