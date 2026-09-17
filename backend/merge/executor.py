"""Merge execution.

The executor walks the projection bottom-up:

* dataset views apply the planned transformations in SQL over the Parquet files;
* **lookups** deduplicate the child on its normalised key and LEFT JOIN it
  onto the parent, so the parent's row count is invariant (no fan-out, no
  silent drops); each join adds a ``_match_*`` flag and ``_src_*`` source
  row ids;
* **aggregations** reduce the child to the key grain first;
* **entity groups** are resolved (Fellegi–Sunter or shared key), then fused
  record-by-record in Python with conflict detection and cell provenance.

Joins run in DuckDB (vectorised, out-of-core); only entity-group records
are materialised in Python. Keys are normalised in SQL for the ``basic`` and
``identifier`` normalisers and through a distinct-value mapping table for the
Python-defined ones, so the join uses exactly the equivalence under which
value overlap was measured.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from backend.conflicts.strategies import Candidate, estimate_source_accuracy, resolve as resolve_conflict
from backend.core.models import Conflict, DatasetArtifact, DatasetProfile, EntityCluster, MergePlan, SemanticType
from backend.core.values import basic_normalize, digits_normalize, get_normalizer
from backend.entity_resolution.comparators import name_compare, normalize_person_name
from backend.entity_resolution.resolver import ERResult, FieldSpec, resolve as resolve_entities
from backend.logging_conf import get_logger
from backend.matching.normalize import name_tokens
from backend.merge.temporal import CREATION_TOKENS, DatedValue, build_intervals
from backend.merge.projection import AttachSpec, ColumnSpec, GroupSpec, NodeSpec, ProjectionBuilder
from backend.merge.transforms import compile_select
from backend.storage.duck import parquet_scan, quote_ident, quote_literal

log = get_logger(__name__)

SQL_NORMALIZERS = {
    "basic": "regexp_replace(lower(trim(CAST({x} AS VARCHAR))), '^(-?[0-9]+)\\.0+$', '\\1')",
    "identity": "regexp_replace(lower(trim(CAST({x} AS VARCHAR))), '^(-?[0-9]+)\\.0+$', '\\1')",
}
_TS_TOKENS = {"updated", "modified", "last", "timestamp", "changed"}


@dataclass
class ExecutionContext:
    merge_id: str
    plan: MergePlan
    spec: NodeSpec
    output_table: str
    conflicts: list[Conflict] = field(default_factory=list)
    cell_provenance: list[dict[str, Any]] = field(default_factory=list)
    er_results: dict[str, ERResult] = field(default_factory=dict)
    entity_tables: dict[str, str] = field(default_factory=dict)
    join_stats: list[dict[str, Any]] = field(default_factory=list)
    group_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    source_accuracy: dict[str, dict[str, float]] = field(default_factory=dict)
    histories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # group id → effective-dated attribute intervals


class MergeExecutor:
    def __init__(self, plan: MergePlan, artifacts: dict[str, DatasetArtifact], profiles: dict[str, DatasetProfile], config: dict, con: duckdb.DuckDBPyConnection | None = None):
        self.plan = plan
        self.artifacts = artifacts
        self.profiles = profiles
        self.config = config
        self.con = con or duckdb.connect(":memory:")
        self.entity_labels: dict[tuple[tuple[str, int], tuple[str, int]], bool] = {}  # user-labelled record pairs (active learning)
        self.calibration_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()  # labelled pairs that were randomly sampled
        engine = config.get("engine", {})
        if engine.get("memory_limit"):
            self.con.execute(f"SET memory_limit={quote_literal(engine['memory_limit'])}")
        if engine.get("temp_directory"):
            Path(engine["temp_directory"]).mkdir(parents=True, exist_ok=True)
            self.con.execute(f"SET temp_directory={quote_literal(Path(engine['temp_directory']).as_posix())}")
        if engine.get("threads"):
            self.con.execute(f"SET threads={int(engine['threads'])}")
        # row order is made explicit with ORDER BY __row where it matters, so DuckDB may stream freely
        self.con.execute("SET preserve_insertion_order=false")
        self._n = 0
        self.vendor = tuple(config["merge"].get("vendor_prefixes", []))

    # ------------------------------------------------------------------ helpers
    def _tmp(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def _columns(self, table: str) -> list[str]:
        return [r[0] for r in self.con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()]

    # ------------------------------------------------------------------ entry point
    def run(self) -> ExecutionContext:
        started = time.perf_counter()
        spec = ProjectionBuilder(self.plan.integration_tree, self.plan.entity_groups, self.profiles, self.plan.transformations, self.vendor).build(self.plan.root_dataset)
        ctx = ExecutionContext(merge_id=f"merge_{uuid.uuid4().hex[:10]}", plan=self.plan, spec=spec, output_table="unified")

        t0 = time.perf_counter()
        for ds in self.plan.datasets:
            self._dataset_view(ds)
        ctx.timings_ms["transform_views"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        table = self._materialize(spec, ctx)
        ctx.timings_ms["joins_and_fusion"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        self._final_projection(spec, table, ctx)
        ctx.timings_ms["projection"] = round((time.perf_counter() - t0) * 1000, 1)
        ctx.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        log.info("Merge planned into views", merge_id=ctx.merge_id, conflicts=len(ctx.conflicts), elapsed_ms=ctx.timings_ms["total"])
        return ctx

    # ------------------------------------------------------------------ dataset views
    def _dataset_view(self, ds: str) -> None:
        art = self.artifacts[ds]
        cols = [c.name for c in self.profiles[ds].columns]
        select, _ = compile_select(cols, self.plan.transformations.get(ds, []))
        src = f"read_parquet({quote_literal(Path(art.storage_path).as_posix())}, file_row_number=true)"
        self.con.execute(
            f"CREATE OR REPLACE TEMP VIEW {quote_ident('v_' + ds)} AS SELECT file_row_number AS __row, {', '.join(select)} FROM {src}"
        )

    # ------------------------------------------------------------------ recursion
    @staticmethod
    def _signature(spec: NodeSpec) -> tuple:
        """Structural identity of a subtree's SQL (independent of output naming)."""
        return (
            spec.dataset,
            tuple((a.kind, tuple(tuple(sorted(k.items())) for k in a.keys), a.alias, MergeExecutor._signature(a.child)) for a in spec.attachments),
            spec.group.group_id if spec.group else None,
        )

    def _materialize(self, spec: NodeSpec, ctx: ExecutionContext) -> str:
        """Materialise a subtree bottom-up. Identical subtrees (the zone dimension used as
        both pickup and dropoff) are computed once and reused."""
        cache = self.__dict__.setdefault("_mat_cache", {})
        sig = self._signature(spec)
        if sig in cache:
            return cache[sig]
        table = self._materialize_uncached(spec, ctx)
        cache[sig] = table
        return table

    def _materialize_uncached(self, spec: NodeSpec, ctx: ExecutionContext) -> str:
        table = f"v_{spec.dataset}"
        for att in spec.attachments:
            child_table = self._materialize(att.child, ctx)
            table = self._attach(table, child_table, att, spec.dataset, ctx)
        if spec.group is not None:
            member_tables = {m: self._materialize(ms, ctx) for m, ms in spec.group.members.items()}
            table = self._fuse(table, member_tables, spec, ctx)
        return table

    # ------------------------------------------------------------------ key normalisation
    def _keyed(self, table: str, column: str, normalizer: str, key_alias: str) -> str:
        """Return a table = ``table`` + normalised key column ``key_alias``."""
        out = self._tmp("k")
        cols = set(self._columns(table))
        if column not in cols and f"{column}_raw" in cols:
            # the key column was replaced by a transformation (e.g. currency text → amount/currency/usd); relationships
            # were measured on the original values, which the view preserves as <column>_raw
            column = f"{column}_raw"
        q = quote_ident(column)
        if normalizer in SQL_NORMALIZERS:
            expr = SQL_NORMALIZERS[normalizer].format(x=q)
            self.con.execute(f"CREATE TEMP VIEW {out} AS SELECT *, {expr} AS {quote_ident(key_alias)} FROM {quote_ident(table)}")
            return out
        fn = get_normalizer(normalizer)
        distinct = [r[0] for r in self.con.execute(f"SELECT DISTINCT CAST({q} AS VARCHAR) FROM {quote_ident(table)} WHERE {q} IS NOT NULL").fetchall()]
        mapping = pa.table({"raw": pa.array(distinct, type=pa.string()), "k": pa.array([fn(v) for v in distinct], type=pa.string())})
        map_name = self._tmp("map")
        self.con.register(map_name + "_arrow", mapping)
        self.con.execute(f"CREATE TEMP TABLE {map_name} AS SELECT * FROM {map_name}_arrow")
        self.con.unregister(map_name + "_arrow")
        self.con.execute(
            f"CREATE TEMP VIEW {out} AS SELECT t.*, m.k AS {quote_ident(key_alias)} FROM {quote_ident(table)} t "
            f"LEFT JOIN {map_name} m ON m.raw = CAST(t.{q} AS VARCHAR)"
        )
        return out

    # ------------------------------------------------------------------ attachments
    def _attach(self, parent: str, child: str, att: AttachSpec, parent_ds: str, ctx: ExecutionContext) -> str:
        key = att.keys[0]
        pk = self._keyed(parent, key["parent_column"], key["normalizer"], "__pk")
        ck = self._keyed(child, key["child_column"], key["normalizer"], "__ck")
        out = self._tmp("j")
        matched_col = next(c for c in att.columns if c.source_column == "__matched__")
        child_cols = set(self._columns(ck))

        if att.kind == "lookup":
            dup = self.con.execute(f"SELECT count(*) - count(DISTINCT __ck) FROM {ck} WHERE __ck IS NOT NULL").fetchone()[0]
            if dup:
                ctx.warnings.append(f"{att.child.dataset}: {dup} rows share a join key with another row; the first occurrence is used for lookups (no parent rows multiplied).")
            dedup = self._tmp("d")
            self.con.execute(f"CREATE TEMP VIEW {dedup} AS SELECT * FROM {ck} WHERE __ck IS NOT NULL QUALIFY row_number() OVER (PARTITION BY __ck ORDER BY __row) = 1")
            sel = [f"c.{quote_ident(c.child_view_name)} AS {quote_ident(c.view_name)}" for c in att.columns if c.child_view_name]
            src_cols = [f"c.__row AS {quote_ident('__src_' + att.alias)}"]
            src_cols += [f"c.{quote_ident(sc)} AS {quote_ident('__src_' + att.alias + '.' + sc[len('__src_'):])}" for sc in sorted(child_cols) if sc.startswith("__src_")]
            sel += src_cols + [f"(c.__row IS NOT NULL) AS {quote_ident(matched_col.view_name)}"]
            parent_cols = [f"p.{quote_ident(c)}" for c in self._columns(parent)]
            self.con.execute(
                f"CREATE TEMP VIEW {out} AS SELECT {', '.join(parent_cols + sel)} FROM {pk} p LEFT JOIN {dedup} c ON p.__pk = c.__ck"
            )
        else:
            aggs = []
            for c in att.columns:
                if c.source_column == "__matched__":
                    continue
                v = quote_ident(c.view_name)
                if c.aggregate == "count":
                    aggs.append(f"count(*) AS {v}")
                    continue
                x = quote_ident(c.child_view_name)
                expr = {
                    "sum": f"sum(TRY_CAST({x} AS DOUBLE))",
                    "avg": f"avg(TRY_CAST({x} AS DOUBLE))",
                    "max": f"max({x})",
                    "mode": f"mode({x})",
                    "count_distinct": f"count(DISTINCT {x})",
                }[c.aggregate]
                # a value constant within the group (e.g. a county figure repeated on every
                # neighbourhood row) is functionally dependent on the key: pass it through
                # instead of summing / counting it once per row
                if c.aggregate in ("sum", "avg", "count_distinct"):
                    expr = f"CASE WHEN count(DISTINCT {x}) <= 1 THEN min(TRY_CAST({x} AS DOUBLE)) ELSE CAST({expr} AS DOUBLE) END"
                else:
                    expr = f"CASE WHEN count(DISTINCT {x}) <= 1 THEN min({x}) ELSE {expr} END"
                aggs.append(f"{expr} AS {v}")
            agg = self._tmp("a")
            self.con.execute(f"CREATE TEMP TABLE {agg} AS SELECT __ck, {', '.join(aggs)} FROM {ck} WHERE __ck IS NOT NULL GROUP BY __ck")
            sel = [f"a.{quote_ident(c.view_name)}" for c in att.columns if c.source_column != "__matched__"]
            sel.append(f"(a.__ck IS NOT NULL) AS {quote_ident(matched_col.view_name)}")
            parent_cols = [f"p.{quote_ident(c)}" for c in self._columns(parent)]
            self.con.execute(f"CREATE TEMP VIEW {out} AS SELECT {', '.join(parent_cols + sel)} FROM {pk} p LEFT JOIN {agg} a ON p.__pk = a.__ck")

        total, matched = self.con.execute(f"SELECT count(*), count(*) FILTER (WHERE {quote_ident(matched_col.view_name)}) FROM {out}").fetchone()
        parent_rows = self.con.execute(f"SELECT count(*) FROM {quote_ident(parent)}").fetchone()[0]
        ctx.join_stats.append({
            "parent": parent_ds, "child": att.child.dataset, "kind": att.kind, "alias": att.alias, "role": att.role,
            "keys": att.keys, "rows": total, "parent_rows_before": parent_rows, "matched_rows": matched, "unmatched_rows": total - matched,
            "match_rate": round(matched / total, 4) if total else 0.0, "confidence": att.confidence, "requires_review": att.requires_review,
        })
        log.info("Join executed", parent=parent_ds, child=att.child.dataset, kind=att.kind, role=att.role, rows=total, matched=matched)
        return out

    # ------------------------------------------------------------------ entity fusion
    def _fuse(self, anchor_table: str, member_tables: dict[str, str], spec: NodeSpec, ctx: ExecutionContext) -> str:
        g: GroupSpec = spec.group  # type: ignore[assignment]
        group = next(x for x in self.plan.entity_groups if x["group_id"] == g.group_id)
        tables = {g.anchor: anchor_table, **member_tables}
        order = [g.anchor] + [d for d in g.datasets if d != g.anchor]
        t0 = time.perf_counter()

        rows: dict[str, list[dict[str, Any]]] = {}
        for ds in order:
            cur = self.con.execute(f"SELECT * FROM {quote_ident(tables[ds])} ORDER BY __row")
            names = [d[0] for d in cur.description]
            rows[ds] = [dict(zip(names, r)) for r in cur.fetchall()]
        index = {ds: {r["__row"]: r for r in rows[ds]} for ds in order}

        fields = [FieldSpec(f["name"], SemanticType(f["semantic_type"]), f["columns"], f["normalizer"]) for f in group["fields"]]
        er: ERResult | None = None
        if group["method"] == "key" or not fields:
            clusters, assignment, link_conf = self._key_groups(order, rows, group)
        else:
            records = {}
            for di, ds in enumerate(order):
                for r in rows[ds]:
                    records[(di, r["__row"])] = {f.name: r.get(f.columns[ds]) for f in fields if ds in f.columns}
            er = resolve_entities({ds: self.artifacts[ds].storage_path for ds in order}, fields, self.config, dedupe=set(order),
                                  auto_threshold=group["auto_threshold"], review_threshold=group["review_threshold"], entity_prefix=f"{g.group_id}-", records=records,
                                  labels=self.entity_labels or None, calibration_pairs=self.calibration_pairs or None)
            ctx.er_results[g.group_id] = er
            clusters, assignment = er.clusters, er.assignment
            link_conf = defaultdict(float)
            for p in er.pairs:
                a, b = (p.left_dataset, p.left_row), (p.right_dataset, p.right_row)
                if assignment.get(a) == assignment.get(b) and p.probability >= group["auto_threshold"]:
                    link_conf[a] = max(link_conf[a], p.probability)
                    link_conf[b] = max(link_conf[b], p.probability)
        t_er = time.perf_counter()

        ts_col = {ds: self._timestamp_column(ds, tables[ds]) for ds in order}
        # history timestamps: change time if available, else creation time
        hist_col = {ds: (ts_col[ds], "change") if ts_col[ds] else (self._timestamp_column(ds, tables[ds], CREATION_TOKENS), "creation") for ds in order}
        history: list[dict[str, Any]] = []
        # descriptive attributes only: identifiers, timestamps and raw-text shadows are not "facts that change"
        history_cols = {c.view_name for c in g.columns if c.members and c.aggregate is None and not c.output.endswith("_raw")
                        and c.semantic_type not in (SemanticType.ID.value, SemanticType.DATE.value, SemanticType.DATETIME.value)}
        priority = {ds: i for i, ds in enumerate(self.plan.overrides.get("source_priority") or order)}
        fused_cols = [c for c in g.columns if c.members]
        field_norm = {f["resolved_output"]: f["normalizer"] for f in group["fields"] if "resolved_output" in f}

        # pass 1: candidate sets per (entity, column); normalisers are memoised per column
        norm_fns = {col.view_name: _memoize(self._comparison_normalizer(col, field_norm.get(col.output))) for col in fused_cols}
        per_entity: list[tuple[EntityCluster, dict[str, list[Candidate]]]] = []
        conflict_sets: list[list[Candidate]] = []
        for cl in clusters:
            cand_map: dict[str, list[Candidate]] = {}
            for col in fused_cols:
                norm_fn = norm_fns[col.view_name]
                cands = []
                for ds, view_col in col.members:
                    for mds, mrow in cl.members:
                        if mds != ds:
                            continue
                        rec = index[ds][mrow]
                        value = rec.get(view_col)
                        if isinstance(value, float) and value != value:
                            value = None
                        ts = rec.get(ts_col[ds]) if ts_col[ds] else None
                        cands.append(Candidate(value, norm_fn(value) if value is not None else None, ds, view_col, mrow,
                                               link_conf.get((ds, mrow), 1.0) if cl.size > 1 else 1.0,
                                               ts if isinstance(ts, datetime) else (datetime(ts.year, ts.month, ts.day) if isinstance(ts, date) else None),
                                               priority.get(ds, 99)))
                if col.semantic_type == SemanticType.NAME.value:
                    _unify_name_variants(cands)
                cand_map[col.view_name] = cands
                if col.view_name in history_cols and len({c.norm for c in cands if c.norm is not None}) > 1 and any(hist_col[c.dataset][0] for c in cands):
                    dated = []
                    for c in cands:
                        hc, kind = hist_col[c.dataset]
                        t = _as_datetime(index[c.dataset][c.row].get(hc)) if hc else None
                        dated.append(DatedValue(c.value, c.norm, c.dataset, c.column, c.row, t, kind if t else None, c.priority))
                    if len({d.norm for d in dated if d.timestamp is not None and d.norm is not None}) > 1:
                        history.extend(build_intervals(cl.entity_id, col.output, dated))
                if col.aggregate is None and len({c.norm for c in cands if c.norm is not None}) > 1:
                    conflict_sets.append(cands)
            per_entity.append((cl, cand_map))
        if self.plan.conflict_strategy == "source_accuracy_vote":
            ctx.source_accuracy[g.group_id] = estimate_source_accuracy(conflict_sets)

        # pass 2: resolve
        out_rows: list[dict[str, Any]] = []
        n_conflicts = 0
        for cl, cand_map in per_entity:
            row: dict[str, Any] = {"__entity_id": cl.entity_id}
            has_conflict = False
            for col in fused_cols:
                cands = cand_map[col.view_name]
                if col.aggregate in ("sum", "count", "count_distinct", "avg", "max"):
                    row[col.view_name] = self._aggregate_values([c.value for c in cands], col.aggregate)
                    continue
                res = resolve_conflict(cands, self.plan.conflict_strategy, ctx.source_accuracy.get(g.group_id))
                row[col.view_name] = res.value
                if res.status in ("flagged", "manual_review"):
                    has_conflict = True
                    n_conflicts += 1
                    ctx.conflicts.append(Conflict(
                        entity_id=cl.entity_id, attribute=col.output, strategy=self.plan.conflict_strategy, reason=res.reason, status=res.status,
                        resolved_value=_json_safe(res.value),
                        candidates=[{"value": _json_safe(c.value), "normalized": c.norm, "source_dataset": c.dataset, "source_column": c.column, "source_row": c.row,
                                     "confidence": round(c.confidence, 4), "record_timestamp": c.timestamp.isoformat() if c.timestamp else None,
                                     "selected": res.winner is not None and c is res.winner} for c in cands],
                    ))
                if res.winner is not None and len(cands) > 1:
                    ctx.cell_provenance.append({"entity_id": cl.entity_id, "attribute": col.output, "source_dataset": res.winner.dataset, "source_column": res.winner.column,
                                                "source_row": res.winner.row, "strategy": self.plan.conflict_strategy if res.status != "resolved" else "agreement", "reason": res.reason})
            extras = {c.source_column: c.view_name for c in g.columns if not c.members}
            row[extras["entity_id"]] = cl.entity_id
            row[extras["_entity_confidence"]] = cl.confidence
            row[extras["_entity_sources"]] = json.dumps([f"{d}:{r}" for d, r in cl.members])
            row[extras["_entity_record_count"]] = cl.size
            row[extras["_has_conflict"]] = has_conflict
            out_rows.append(row)

        ent_table = f"entities_{g.group_id}"
        entity_cols = ["__entity_id"] + [c.view_name for c in g.columns]
        arrow = _rows_to_arrow(out_rows, entity_cols)
        self.con.register("__ent_arrow", arrow)
        self.con.execute(f"CREATE OR REPLACE TABLE {ent_table} AS SELECT * FROM __ent_arrow")
        self.con.unregister("__ent_arrow")
        ctx.entity_tables[g.group_id] = ent_table
        if history:
            for h in history:
                h["value"] = _json_safe(h["value"])
            ctx.histories[g.group_id] = history

        member_records = sum(len(rows[d]) for d in order)
        within_dups = 0
        for c in clusters:
            per_ds: dict[str, int] = defaultdict(int)
            for d, _ in c.members:
                per_ds[d] += 1
            within_dups += sum(n - 1 for n in per_ds.values())
        ctx.group_stats[g.group_id] = {
            "datasets": order, "anchor": g.anchor, "method": group["method"], "records": member_records, "entities": len(clusters),
            "multi_source_entities": sum(1 for c in clusters if len({d for d, _ in c.members}) > 1),
            "within_dataset_duplicates_resolved": within_dups,
            "cross_source_links": member_records - len(clusters) - within_dups,
            "duplicates_resolved": within_dups, "conflicts": n_conflicts,
            "entity_resolution": er.stats if er else {"method": "key"},
            "fellegi_sunter": er.model.to_dict() if er else None,
            "elapsed_ms": {"resolution": round((t_er - t0) * 1000, 1), "fusion": round((time.perf_counter() - t_er) * 1000, 1)},
        }
        log.info("Entity fusion completed", group=g.group_id, records=member_records, entities=len(clusters), conflicts=n_conflicts)
        if n_conflicts:
            log.warning("Conflicting field detected", group=g.group_id, conflicts=n_conflicts, strategy=self.plan.conflict_strategy)

        if spec.entity_grain:
            return ent_table
        # anchor rows keep their grain: each row gets its entity's fused attributes
        amap = pa.table({"__row": pa.array([r for (d, r) in assignment if d == g.anchor], type=pa.int64()),
                         "__entity_id": pa.array([e for (d, _), e in assignment.items() if d == g.anchor], type=pa.string())})
        amap_table = self._tmp("amap")
        self.con.register("__amap_arrow", amap)
        self.con.execute(f"CREATE TEMP TABLE {amap_table} AS SELECT * FROM __amap_arrow")
        self.con.unregister("__amap_arrow")
        out = self._tmp("f")
        ent_sel = [f"e.{quote_ident(c.view_name)}" for c in g.columns]
        anchor_cols = [f"a.{quote_ident(c)}" for c in self._columns(anchor_table)]
        self.con.execute(
            f"CREATE TEMP VIEW {out} AS SELECT {', '.join(anchor_cols + ent_sel)} FROM {quote_ident(anchor_table)} a "
            f"JOIN {amap_table} m ON m.__row = a.__row JOIN {ent_table} e ON e.__entity_id = m.__entity_id"
        )
        return out

    def _key_groups(self, order: list[str], rows: dict[str, list[dict]], group: dict) -> tuple[list[EntityCluster], dict, dict]:
        """Deterministic grouping on shared identifier fields (union-find over equal normalised keys)."""
        id_fields = [f for f in group["fields"] if f["semantic_type"] in (SemanticType.ID.value, SemanticType.ZIPCODE.value)] or group["fields"][:1]
        parent: dict[tuple[str, int], tuple[str, int]] = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        first_by_key: dict[tuple[str, str], tuple[str, int]] = {}
        for ds in order:
            for r in rows[ds]:
                node = (ds, r["__row"])
                parent[node] = node
                for f in id_fields:
                    col = f["columns"].get(ds)
                    if col is None:
                        continue
                    k = get_normalizer(f["normalizer"])(r.get(col))
                    if k is None:
                        continue
                    key = (f["name"], k)
                    if key in first_by_key:
                        parent[find(node)] = find(first_by_key[key])
                    else:
                        first_by_key[key] = node
        comps: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        for node in parent:
            comps[find(node)].append(node)
        clusters, assignment = [], {}
        for i, members in enumerate(sorted(comps.values(), key=lambda m: min(m))):
            eid = f"{group['group_id']}-{i:07d}"
            members.sort()
            for m in members:
                assignment[m] = eid
            clusters.append(EntityCluster(entity_id=eid, members=members, confidence=1.0, size=len(members)))
        return clusters, assignment, {}

    def _timestamp_column(self, ds: str, table: str, tokens: set[str] | None = None) -> str | None:
        cols = set(self._columns(table))
        for c in self.profiles[ds].columns:
            if c.name in cols and c.semantic_type in (SemanticType.DATETIME, SemanticType.DATE) and set(name_tokens(c.name)) & (tokens or _TS_TOKENS):
                return c.name
        return None

    @staticmethod
    def _comparison_normalizer(col: ColumnSpec, field_normalizer: str | None):
        st = col.semantic_type
        if st == SemanticType.NAME.value:
            def name_norm(v):
                n = normalize_person_name(v)
                return " ".join(sorted(t for t in n.split() if len(t) > 1)) if n else None
            return name_norm
        if st == SemanticType.PHONE.value:
            return digits_normalize
        if st in (SemanticType.DATE.value, SemanticType.DATETIME.value):
            return lambda v: v.isoformat() if hasattr(v, "isoformat") else basic_normalize(v)
        if field_normalizer:
            return get_normalizer(field_normalizer)
        return basic_normalize

    @staticmethod
    def _aggregate_values(values: list[Any], agg: str) -> Any:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        if agg in ("sum", "count", "count_distinct"):
            return sum(float(v) for v in vals)
        if agg == "avg":
            return sum(float(v) for v in vals) / len(vals)
        return max(vals)

    # ------------------------------------------------------------------ final output
    def _final_projection(self, spec: NodeSpec, table: str, ctx: ExecutionContext) -> None:
        cols = set(self._columns(table))
        if spec.entity_grain:
            record_id = "__entity_id"
            select = [f"{quote_ident(record_id)} AS _record_id"]
        else:
            select = [f"{quote_literal(spec.dataset + ':')} || CAST(__row AS VARCHAR) AS _record_id", f"__row AS {quote_ident('_src_' + spec.dataset + '_row')}"]
        for c in spec.columns:
            if c.view_name in cols:
                select.append(f"{quote_ident(c.view_name)} AS {quote_ident(c.output)}")
        for sc in sorted(c for c in cols if c.startswith("__src_")):
            select.append(f"{quote_ident(sc)} AS {quote_ident('_src_' + sc[len('__src_'):].replace('.', '__') + '_row')}")
        conf_terms = [f"CASE WHEN {quote_ident(c.view_name)} THEN {att.confidence!r} END" for att in spec.attachments for c in att.columns if c.source_column == "__matched__" and c.view_name in cols]
        if conf_terms:
            select.append(f"CAST(list_min([{', '.join(conf_terms)}]) AS DOUBLE) AS _match_confidence")
        link_terms, rel_terms = [], []
        for v, conf in _probability_terms(spec):
            if v not in cols:
                continue
            if conf is None:
                link_terms.append(f"COALESCE(TRY_CAST({quote_ident(v)} AS DOUBLE), 1.0)")
            else:
                rel_terms.append(f"CASE WHEN {quote_ident(v)} THEN {conf!r} ELSE 1.0 END")
        # record-level link uncertainty (independent across rows) vs schema-level relationship confidence (shared by all rows)
        select.append(f"CAST({' * '.join(link_terms) if link_terms else '1.0'} AS DOUBLE) AS _match_probability")
        if rel_terms:
            select.append(f"CAST({' * '.join(rel_terms)} AS DOUBLE) AS _relationship_confidence")
        order = "" if spec.entity_grain else " ORDER BY __row"
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW unified AS SELECT {', '.join(select)} FROM {quote_ident(table)}{order}")

    def materialize_output(self, path: Path) -> None:
        """Stream the unified view to Parquet (DuckDB spills to disk if needed), then serve it from the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY unified TO {quote_literal(path.as_posix())} (FORMAT parquet, COMPRESSION zstd)")
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW unified AS SELECT * FROM read_parquet({quote_literal(path.as_posix())})")


def _memoize(fn):
    cache: dict[Any, Any] = {}

    def wrapped(v):
        key = v if isinstance(v, (str, int, float, bool)) else repr(v)
        if key not in cache:
            cache[key] = fn(v)
        return cache[key]

    return wrapped


def _unify_name_variants(cands: list[Candidate]) -> None:
    """'Rahul Sharma', 'RAHUL K SHARMA' and 'R. Sharma' are representation variants,
    not factual conflicts: if every pair is name-compatible, give them one
    normalised form (that of the most informative value)."""
    present = [c for c in cands if c.norm is not None]
    if len({c.norm for c in present}) < 2:
        return
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            if name_compare(a.value, b.value)[0] not in ("exact", "high", "medium"):
                return
    richest = max(present, key=lambda c: (len(c.norm or ""), -c.priority))
    for c in present:
        c.norm = richest.norm


def _probability_terms(node: NodeSpec) -> list[tuple[str, float | None]]:
    """Uncertainty factors of an output row, as columns of ``node``'s table.

    (probability column, None): record-level — the fused entity's link probability
        (Fellegi–Sunter cluster confidence). Independent across rows.
    (flag column, confidence): schema-level — the lookup matched, so the row depends on that
        relationship being right. Shared by every row using the relationship (fully correlated),
        so it must not be treated as a per-row Bernoulli probability.
    Nested lookups are followed through the columns their parent attachment passes up.
    Unmatched lookups contribute nothing: no link was asserted, so no link can be wrong.
    """
    terms: list[tuple[str, float | None]] = []
    if node.group is not None:
        terms += [(c.view_name, None) for c in node.group.columns if c.source_column == "_entity_confidence"]
    for att in node.attachments:
        terms += [(c.view_name, att.confidence) for c in att.columns if c.source_column == "__matched__"]
        if att.kind != "lookup":
            continue
        passed_up = {c.child_view_name: c.view_name for c in att.columns if c.child_view_name}
        terms += [(passed_up[v], conf) for v, conf in _probability_terms(att.child) if v in passed_up]
    return terms


def relationship_flags(spec: NodeSpec) -> list[dict[str, Any]]:
    """Every join a unified row can depend on: its match-flag output column, parent, child, role, kind, confidence."""
    out_name = {c.view_name: c.output for c in spec.columns}

    def walk(node: NodeSpec, chain: list[list[Any]]) -> list[tuple[str, dict[str, Any]]]:
        found: list[tuple[str, dict[str, Any]]] = []
        for att in node.attachments:
            here = chain + [[att.child.dataset, att.role]]
            info = {"parent": node.dataset, "child": att.child.dataset, "role": att.role, "kind": att.kind, "confidence": att.confidence, "chain": here,
                    # row lineage column that is non-null exactly when this (top-level) join matched
                    "src_row_column": f"_src_{att.alias}_row" if not chain else None}
            # the attachment's own flag (passed-up flags of nested joins also carry source "__matched__" but have a child view)
            found += [(c.view_name, info) for c in att.columns if c.source_column == "__matched__" and c.child_view_name is None]
            if att.kind != "lookup":
                continue
            passed_up = {c.child_view_name: c.view_name for c in att.columns if c.child_view_name}
            found += [(passed_up[v], i) for v, i in walk(att.child, here) if v in passed_up]
        return found

    return [{**info, "flag_column": out_name.get(v)} for v, info in walk(spec, []) if v in out_name or info["src_row_column"]]


def _as_datetime(ts: Any) -> datetime | None:
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None) if ts.tzinfo else ts
    if isinstance(ts, date):
        return datetime(ts.year, ts.month, ts.day)
    return None


def _json_safe(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, float) and v != v:
        return None
    return v


def _rows_to_arrow(rows: list[dict[str, Any]], columns: list[str]) -> pa.Table:
    arrays = {}
    for c in columns:
        vals = [r.get(c) for r in rows]
        try:
            arr = pa.array(vals)
            if pa.types.is_null(arr.type):
                arr = pa.array(vals, type=pa.string())
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arr = pa.array([None if v is None else str(v) for v in vals], type=pa.string())
        arrays[c] = arr
    return pa.table(arrays)
