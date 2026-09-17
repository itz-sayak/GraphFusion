"""Static projection of a merge plan onto an output schema.

The projection is computed purely from the plan and the profiles (no data is
read), so the planner can show the exact unified schema — with lineage —
before anything executes, and the executor builds precisely that schema.

Structure (mirrors the integration tree, rooted at the finest grain):

    NodeSpec(dataset)
      ├── own columns (after transformations), renamed to canonical snake_case
      ├── LookupSpec(child NodeSpec, keys, role)      many-to-one enrichment
      ├── AggregateSpec(child NodeSpec, keys, role)   child aggregated to the join key first
      └── GroupSpec (when the dataset anchors an entity group)
              fused canonical attributes across all group members
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.core.models import DataType, DatasetProfile, SemanticType
from backend.matching.normalize import name_tokens
from backend.merge.canonical import NameAllocator, canonical_attribute_name, role_for, snake

SUM_TOKENS = {"count", "total", "population", "number", "sum", "amount", "spent", "quantity", "fare", "tip", "revenue", "units", "trips"}
AVG_TOKENS = {"median", "mean", "average", "rate", "percent", "ratio", "income", "per", "share", "index", "age"}
META_COLUMNS = ("_record_id",)


@dataclass
class ColumnSpec:
    output: str
    source_dataset: str
    source_column: str  # column in the (transformed) dataset view
    transformation: str | None = None
    operation: str = "select"  # select | lookup | aggregate | fuse
    semantic_type: str = SemanticType.UNKNOWN.value
    data_type: str = DataType.STRING.value
    aggregate: str | None = None
    members: list[tuple[str, str]] = field(default_factory=list)  # fused: (dataset, column name in that dataset's enriched view)
    view_name: str = ""  # name inside this node's enriched SQL view (raw column or internal alias)
    child_view_name: str | None = None  # lookup/aggregate: the column in the child's enriched view


@dataclass
class AttachSpec:
    kind: str  # lookup | aggregate
    child: "NodeSpec"
    keys: list[dict[str, str]]  # parent_column (view col), child_column (view col), normalizer
    role: str | None
    alias: str
    confidence: float
    columns: list[ColumnSpec] = field(default_factory=list)  # outputs added to the parent
    renamed_parent_keys: dict[str, str] = field(default_factory=dict)
    requires_review: bool = False


@dataclass
class GroupSpec:
    group_id: str
    anchor: str
    datasets: list[str]
    method: str
    fields: list[dict[str, Any]]
    members: dict[str, "NodeSpec"] = field(default_factory=dict)  # non-anchor members (with their own attachments)
    columns: list[ColumnSpec] = field(default_factory=list)


@dataclass
class NodeSpec:
    dataset: str
    view_columns: list[str]
    columns: list[ColumnSpec] = field(default_factory=list)  # final outputs of this node's subtree
    attachments: list[AttachSpec] = field(default_factory=list)
    group: GroupSpec | None = None
    excluded_columns: set[str] = field(default_factory=set)
    entity_grain: bool = False  # root anchor of an entity group: one output row per entity


def view_columns(profile: DatasetProfile, transforms: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict]]:
    by_col = {t["column"]: t for t in transforms}
    cols: list[str] = []
    info: dict[str, dict] = {}
    for c in profile.columns:
        t = by_col.get(c.name)
        base = {"semantic_type": c.semantic_type.value, "data_type": c.data_type.value, "source_column": c.name}
        if t is None:
            cols.append(c.name)
            info[c.name] = {**base, "transformation": None}
        elif t["transform"] == "parse_date":
            cols += [c.name, c.name + "_raw"]
            info[c.name] = {**base, "transformation": "parse_date", "data_type": (DataType.DATETIME if t["has_time"] else DataType.DATE).value}
            info[c.name + "_raw"] = {**base, "transformation": "raw_copy"}
        elif t["transform"] == "parse_currency":
            a, cur, conv = t["outputs"]
            cols += [a, cur, conv, c.name + "_raw"]
            info[a] = {**base, "transformation": "parse_currency.amount", "data_type": DataType.FLOAT.value}
            info[cur] = {**base, "transformation": "parse_currency.currency", "semantic_type": SemanticType.CURRENCY_CODE.value}
            info[conv] = {**base, "transformation": f"parse_currency.convert_to_{t['base_currency']}", "data_type": DataType.FLOAT.value}
            info[c.name + "_raw"] = {**base, "transformation": "raw_copy"}
        elif t["transform"] == "convert_currency":
            (conv,) = t["outputs"]
            cols += [c.name, conv]
            info[c.name] = {**base, "transformation": None}
            info[conv] = {**base, "transformation": f"convert_currency_to_{t['base_currency']}", "data_type": DataType.FLOAT.value}
    return cols, info


def aggregate_function(column: str, data_type: str, semantic_type: str) -> str | None:
    toks = set(name_tokens(column))
    if data_type in (DataType.INTEGER.value, DataType.FLOAT.value):
        if semantic_type == SemanticType.ID.value:
            return "count_distinct"
        if toks & AVG_TOKENS:
            return "avg"
        if toks & SUM_TOKENS or semantic_type == SemanticType.CURRENCY.value:
            return "sum"
        return "avg"
    if data_type in (DataType.DATE.value, DataType.DATETIME.value):
        return "max"
    if semantic_type in (SemanticType.CATEGORY.value, SemanticType.CITY.value, SemanticType.COUNTRY.value, SemanticType.CURRENCY_CODE.value):
        return "mode"
    return None


class ProjectionBuilder:
    def __init__(self, plan_tree: list[dict[str, Any]], groups: list[dict[str, Any]], profiles: dict[str, DatasetProfile], transforms: dict[str, list[dict]], vendor_prefixes: tuple[str, ...]):
        self.tree = plan_tree
        self.groups = {g["group_id"]: g for g in groups}
        self.group_of = {ds: g["group_id"] for g in groups for ds in g["datasets"]}
        self.profiles = profiles
        self.transforms = transforms
        self.vendor = vendor_prefixes
        self._counter = 0
        self.children: dict[str, list[dict[str, Any]]] = {}
        for e in plan_tree:
            self.children.setdefault(e["parent"], []).append(e)

    def _internal(self) -> str:
        self._counter += 1
        return f"__c{self._counter}"

    def build(self, root: str) -> NodeSpec:
        return self._node(root, priority=[root], is_root=True)

    # ------------------------------------------------------------------
    def _node(self, ds: str, priority: list[str], excluded: set[str] | None = None, is_root: bool = False) -> NodeSpec:
        # structurally identical subtrees (a dimension referenced in several roles) share one
        # spec, so the executor computes them once and internal column names line up
        key = (ds, frozenset(excluded or ()), is_root)
        cache = self.__dict__.setdefault("_spec_cache", {})
        if key not in cache:
            cache[key] = self._build_node(ds, priority, excluded, is_root)
        return cache[key]

    def _build_node(self, ds: str, priority: list[str], excluded: set[str] | None = None, is_root: bool = False) -> NodeSpec:
        cols, info = view_columns(self.profiles[ds], self.transforms.get(ds, []))
        spec = NodeSpec(dataset=ds, view_columns=cols, excluded_columns=set(excluded or ()))
        alloc = NameAllocator(set(META_COLUMNS))
        gid = self.group_of.get(ds)
        group = self.groups.get(gid) if gid else None
        is_anchor = group is not None and group["anchor"] == ds

        fused_cols: set[tuple[str, str]] = set()
        if is_anchor:
            for f in group["fields"]:
                fused_cols |= set(f["columns"].items())
        edges = [e for e in self.children.get(ds, []) if e["join_kind"] in ("lookup", "aggregate_lookup", "reverse_aggregate")]
        renames: dict[str, str] = {}
        for e in edges:
            for k in e["key_pairs"]:
                if e["join_kind"] == "lookup" and k.get("role"):
                    renames[k["parent_column"]] = f"{k['role']}_{snake(k['child_column'], self.vendor)}"

        # 1) own columns -- matched attributes of an anchor are replaced by fused values
        own: list[ColumnSpec] = []
        for c in cols:
            if c in spec.excluded_columns or (is_anchor and (ds, c) in fused_cols):
                continue
            out = alloc.allocate(renames.get(c) or snake(c, self.vendor), ds)
            own.append(ColumnSpec(out, ds, c, info[c]["transformation"], "select", info[c]["semantic_type"], info[c]["data_type"], view_name=c))

        # 2) attachments to children outside any entity group
        attached: list[ColumnSpec] = []
        for e in edges:
            child = e["child"]
            child_keys = {k["child_column"] for k in e["key_pairs"]}
            if e["join_kind"] == "lookup":
                # one attachment per role (PULocationID and DOLocationID both reference LocationID)
                for k in e["key_pairs"]:
                    child_spec = self._node(child, priority=priority + [child], excluded={k["child_column"]})
                    role = k.get("role")
                    alias = f"{role}_{child}" if role else child
                    att = AttachSpec("lookup", child_spec, [k], role, alias, e["confidence"], requires_review=e.get("requires_review", False))
                    for c in child_spec.columns:
                        desired = c.output
                        if role:
                            desired = f"_{role}{c.output}" if c.output.startswith("_match_") or c.output.startswith("_") else f"{role}_{c.output}"
                        out = alloc.allocate(desired, child)
                        att.columns.append(ColumnSpec(out, c.source_dataset, c.source_column, c.transformation, "lookup", c.semantic_type, c.data_type,
                                                      members=c.members, view_name=self._internal(), child_view_name=c.view_name))
                    att.columns.append(ColumnSpec(alloc.allocate(f"_match_{alias}", ds), child, "__matched__", None, "lookup", SemanticType.BOOLEAN.value, DataType.BOOLEAN.value, view_name=self._internal()))
                    spec.attachments.append(att)
                    attached.extend(att.columns)
            else:  # aggregate: names are <column>_<aggregate>; collisions get the child qualifier
                child_spec = self._node(child, priority=priority + [child])
                role = e["key_pairs"][0].get("role") if e["key_pairs"] else None
                alias = f"{role}_{child}" if role else child
                att = AttachSpec("aggregate", child_spec, e["key_pairs"], role, alias, e["confidence"], requires_review=e.get("requires_review", False))
                for c in child_spec.columns:
                    if c.source_dataset == child and c.source_column in child_keys:
                        continue
                    agg = aggregate_function(c.source_column, c.data_type, c.semantic_type)
                    if agg is None:
                        continue
                    dtype = DataType.FLOAT.value if agg in ("avg", "sum") else (DataType.INTEGER.value if agg == "count_distinct" else c.data_type)
                    suffix = "" if agg in ("mode", "max") else f"_{agg}"
                    out = alloc.allocate(f"{c.output}{suffix}" if not c.output.startswith("_") else f"_{child}{c.output}", child)
                    att.columns.append(ColumnSpec(out, c.source_dataset, c.source_column, c.transformation, "aggregate", c.semantic_type, dtype, aggregate=agg,
                                                  view_name=self._internal(), child_view_name=c.view_name))
                att.columns.append(ColumnSpec(alloc.allocate(f"{alias}_record_count", child), child, "__count__", None, "aggregate", SemanticType.NUMERIC.value, DataType.INTEGER.value, aggregate="count", view_name=self._internal()))
                att.columns.append(ColumnSpec(alloc.allocate(f"_match_{alias}", ds), child, "__matched__", None, "aggregate", SemanticType.BOOLEAN.value, DataType.BOOLEAN.value, view_name=self._internal()))
                spec.attachments.append(att)
                attached.extend(att.columns)

        # 3) entity-group fusion at the anchor
        group_cols: list[ColumnSpec] = []
        if is_anchor:
            gspec = GroupSpec(group["group_id"], ds, group["datasets"], group["method"], group["fields"])
            member_order = [ds] + [d for d in group["datasets"] if d != ds]
            group_cols.append(ColumnSpec(alloc.allocate("entity_id", ds), ds, "entity_id", None, "fuse", SemanticType.ID.value, DataType.STRING.value, view_name=self._internal()))
            for f in group["fields"]:
                out = alloc.allocate(f["output_name"], group["group_id"])
                f_members = sorted(f["columns"].items(), key=lambda kv: member_order.index(kv[0]))
                group_cols.append(ColumnSpec(out, f_members[0][0], f_members[0][1], None, "fuse", f["semantic_type"], f.get("data_type", DataType.STRING.value),
                                             members=f_members, view_name=self._internal()))
                f["resolved_output"] = out
            if is_root:
                # entity grain: the anchor's per-row columns are fused per entity as well
                for c in own + attached:
                    agg = c.aggregate if c.operation == "aggregate" else None
                    group_cols.append(ColumnSpec(c.output, c.source_dataset, c.source_column, c.transformation, "fuse", c.semantic_type, c.data_type,
                                                 aggregate=agg, members=[(ds, c.view_name)], view_name=self._internal()))
            for m in member_order[1:]:
                member = self._node(m, priority=priority + [m], excluded={c for d, c in fused_cols if d == m})
                gspec.members[m] = member
                for c in member.columns:
                    out = alloc.allocate(c.output, m)
                    group_cols.append(ColumnSpec(out, c.source_dataset, c.source_column, c.transformation, "fuse", c.semantic_type, c.data_type,
                                                 aggregate=c.aggregate if c.operation == "aggregate" else None, members=[(m, c.view_name)], view_name=self._internal()))
            for extra, dt in (("_entity_confidence", DataType.FLOAT.value), ("_entity_sources", DataType.STRING.value), ("_entity_record_count", DataType.INTEGER.value), ("_has_conflict", DataType.BOOLEAN.value)):
                group_cols.append(ColumnSpec(alloc.allocate(extra, ds), ds, extra, None, "fuse", SemanticType.UNKNOWN.value, dt, view_name=self._internal()))
            gspec.columns = group_cols
            spec.group = gspec

        spec.entity_grain = is_anchor and is_root
        spec.columns = group_cols if spec.entity_grain else own + group_cols + attached
        return spec


def flatten_schema(spec: NodeSpec) -> list[ColumnSpec]:
    return list(spec.columns)
