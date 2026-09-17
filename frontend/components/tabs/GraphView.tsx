"use client";

import { memo, useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, { Background, Controls, Edge, Handle, MarkerType, MiniMap, Node, NodeProps, Position } from "reactflow";
import "reactflow/dist/style.css";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { MatchExplanation, SchemaColumn, SchemaLink, SchemaMap, SchemaRelationship, SchemaTable } from "@/lib/types";
import EvidencePanel from "../EvidencePanel";
import { Badge, Button, Empty, ErrorNote, Spinner, fmt } from "../ui";

/* ------------------------------------------------------------------ visual encoding
 * Colour = relationship family (validated categorical slots for the dark surface, all-pairs CVD-safe):
 *   reference (lookup / aggregate lookup)  blue    · same entity by shared key  aqua · probabilistic entity match  orange
 * Line style = how strong the link is:  join key  solid (aggregated lookups dashed) · shared attribute  dashed grey · candidate  dotted grey
 * Every edge also carries a text label or tooltip, so identity never relies on colour alone. */
const FAMILY_COLOR: Record<string, string> = { lookup: "#3987e5", aggregate_lookup: "#3987e5", entity_key_merge: "#199e70", entity_resolution: "#d95926" };
const NEUTRAL = "#8b94c7";
const KIND_LABEL: Record<string, string> = {
  lookup: "lookup (many → one)",
  aggregate_lookup: "aggregated lookup",
  entity_key_merge: "same entity (shared key)",
  entity_resolution: "same entity (probabilistic)",
};
const ROLE_ICON: Record<SchemaColumn["role"], { icon: string; title: string }> = {
  key: { icon: "🔑", title: "Key: unique identifier of this table" },
  reference: { icon: "↗", title: "Reference: points to a key in another table" },
  linked: { icon: "⇄", title: "Linked: the same information exists in another table" },
  attribute: { icon: "·", title: "Not connected to other tables" },
};

const TABLE_W = 300;
const HEADER_H = 58;
const ROW_H = 26;
const FOOTER_H = 26;
const LAYER_GAP = 150;
const ROW_GAP = 36;
const BLOCK_GAP = 90;

type Mode = "columns" | "datasets";

interface Filters {
  joinKeys: boolean;
  attributes: boolean;
  candidates: boolean;
  minConfidence: number;
  mergeOnly: boolean;
  allColumns: boolean;
  search: string;
}

type Selection =
  | { type: "column"; dataset: string; column: string }
  | { type: "link"; id: string }
  | { type: "table"; id: string }
  | { type: "relationship"; id: string }
  | null;

const colKey = (ds: string, col: string) => `${ds}\u0000${col}`;
const handleId = (side: "l" | "r", kind: "s" | "t", col: string) => `${side}${kind}:${col}`;

/** Percent that never rounds a small non-zero share down to "0%". */
function pct(x: number): string {
  if (x > 0 && x < 0.01) return "<1%";
  if (x < 1 && x > 0.99) return ">99%";
  return `${Math.round(x * 100)}%`;
}

function linkColor(l: SchemaLink): string {
  return l.kind === "join_key" && l.join_kind ? FAMILY_COLOR[l.join_kind] || NEUTRAL : NEUTRAL;
}

/* ------------------------------------------------------------------ nodes */
interface TableData {
  table: SchemaTable;
  columns: SchemaColumn[];
  hidden: number;
  mode: Mode;
  focus: "none" | "focus" | "dim";
  highlightCols: Set<string>;
  handleSides: Map<string, Set<"l" | "r">>;
  onColumn: (ds: string, col: string) => void;
  onTable: (id: string) => void;
  onToggle: (id: string) => void;
  expanded: boolean;
  searchHit: (text: string) => boolean;
}

const TableNode = memo(function TableNode({ data }: NodeProps<TableData>) {
  const { table: t, columns, hidden, mode, focus, highlightCols, handleSides } = data;
  return (
    <div
      className={`rounded-xl border bg-ink-900 shadow-xl transition-opacity ${focus === "focus" ? "border-accent-400" : "border-ink-600"} ${focus === "dim" ? "opacity-30" : ""}`}
      style={{ width: TABLE_W }}
      data-testid="schema-table"
    >
      <button
        className={`nodrag flex w-full flex-col items-start rounded-t-xl border-b border-ink-700 px-3 py-2 text-left ${data.searchHit(t.label) ? "bg-accent-500/20" : "bg-ink-800"}`}
        style={{ height: HEADER_H }}
        onClick={() => data.onTable(t.id)}
        title={`${t.label} — click for its relationships`}
      >
        <div className="flex w-full items-center justify-between gap-2">
          <span className="truncate text-[13px] font-semibold text-white">{t.label}</span>
          <span className="shrink-0 rounded bg-ink-700 px-1.5 text-[10px] uppercase text-ink-300">{t.source_type}</span>
        </div>
        <div className="text-[11px] text-ink-400">
          {fmt(t.rows ?? 0)} rows · {t.column_count} columns · {t.connected_columns} connected
        </div>
      </button>
      {mode === "datasets" && (
        <>
          <Handle type="source" position={Position.Right} id="rs:__table" className="!h-2 !w-2 !border-0 !bg-ink-400" />
          <Handle type="target" position={Position.Left} id="lt:__table" className="!h-2 !w-2 !border-0 !bg-ink-400" />
          <Handle type="source" position={Position.Left} id="ls:__table" className="!opacity-0" />
          <Handle type="target" position={Position.Right} id="rt:__table" className="!opacity-0" />
        </>
      )}
      {mode === "columns" &&
        columns.map((c) => {
          const hl = highlightCols.has(c.name);
          const sides = handleSides.get(c.name);
          const role = ROLE_ICON[c.role];
          return (
            <div
              key={c.name}
              className={`nodrag relative flex cursor-pointer items-center gap-2 px-3 text-[12px] ${hl ? "bg-accent-500/25" : data.searchHit(c.name) ? "bg-amber-500/15" : "hover:bg-ink-800"}`}
              style={{ height: ROW_H }}
              onClick={() => data.onColumn(t.id, c.name)}
              title={`${c.name}${c.description ? ` — ${c.description}` : ""}\n${role.title}\n${c.semantic_type} · ${c.data_type} · ${pct(c.uniqueness)} unique · ${pct(c.null_pct)} null`}
              data-testid="schema-column"
            >
              {sides?.has("l") && (
                <>
                  <Handle type="target" position={Position.Left} id={handleId("l", "t", c.name)} className="!h-2 !w-2 !border-0 !bg-ink-300" />
                  <Handle type="source" position={Position.Left} id={handleId("l", "s", c.name)} className="!h-2 !w-2 !border-0 !bg-ink-300" />
                </>
              )}
              <span className="w-4 shrink-0 text-center text-[11px]" aria-label={c.role}>{role.icon}</span>
              <span className={`min-w-0 flex-1 truncate font-mono ${c.connected ? "text-white" : "text-ink-400"}`}>
                {c.name}
                {c.description && <span className="ml-1 font-sans text-[10.5px] text-ink-400">({c.description.replace(/_/g, " ")})</span>}
              </span>
              <span className="shrink-0 text-[10px] text-ink-400">{c.semantic_type.toLowerCase()}</span>
              {sides?.has("r") && (
                <>
                  <Handle type="source" position={Position.Right} id={handleId("r", "s", c.name)} className="!h-2 !w-2 !border-0 !bg-ink-300" />
                  <Handle type="target" position={Position.Right} id={handleId("r", "t", c.name)} className="!h-2 !w-2 !border-0 !bg-ink-300" />
                </>
              )}
            </div>
          );
        })}
      {mode === "columns" && (hidden > 0 || data.expanded) && (
        <button className="nodrag w-full rounded-b-xl px-3 text-left text-[11px] text-accent-300 hover:underline" style={{ height: FOOTER_H }} onClick={() => data.onToggle(t.id)}>
          {data.expanded ? "hide unconnected columns" : `+ ${hidden} more column${hidden === 1 ? "" : "s"}`}
        </button>
      )}
    </div>
  );
});

function BlockLabel({ data }: NodeProps<{ label: string }>) {
  return <div className="text-[12px] font-semibold uppercase tracking-wide text-ink-400">{data.label}</div>;
}

const nodeTypes = { table: TableNode, label: BlockLabel };

/* ------------------------------------------------------------------ component */
export default function GraphView() {
  const { session, version, sendToChat } = useStore();
  const [map, setMap] = useState<SchemaMap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<Mode>("columns");
  const [filters, setFilters] = useState<Filters>({ joinKeys: true, attributes: true, candidates: false, minConfidence: 0, mergeOnly: false, allColumns: false, search: "" });
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selection, setSelection] = useState<Selection>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<MatchExplanation | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setFullscreen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const canDiscover = (session?.datasets.length || 0) >= 2;

  useEffect(() => {
    if (!canDiscover) {
      setMap(null);
      return;
    }
    setLoading(true);
    setError(null);
    api
      .get<SchemaMap>("/integration/graph?level=schema")
      .then((m) => {
        setMap(m);
        setSelection(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [version, canDiscover]);

  const tables = useMemo(() => new Map((map?.tables || []).map((t) => [t.id, t])), [map]);
  const layerOf = useCallback((ds: string) => tables.get(ds)?.layer ?? 0, [tables]);

  // links that pass the filters
  const visibleLinks = useMemo(() => {
    if (!map) return [] as SchemaLink[];
    return map.links.filter((l) => {
      if (l.kind === "join_key" && !filters.joinKeys) return false;
      if (l.kind === "attribute" && !filters.attributes) return false;
      if ((l.kind === "candidate" || l.kind === "rejected") && !filters.candidates) return false;
      if (l.confidence < filters.minConfidence) return false;
      if (filters.mergeOnly && !l.in_merge_tree) return false;
      return true;
    });
  }, [map, filters]);

  const visibleRels = useMemo(
    () => (map?.relationships || []).filter((r) => r.confidence >= filters.minConfidence && (!filters.mergeOnly || r.in_merge_tree)),
    [map, filters],
  );

  // what the current selection / hover relates to
  const focus = useMemo(() => {
    const cols = new Set<string>();
    const tbls = new Set<string>();
    const lks = new Set<string>();
    const target = selection ?? (hovered ? ({ type: "table", id: hovered } as const) : null);
    if (!target) return null;
    if (target.type === "column") {
      cols.add(colKey(target.dataset, target.column));
      tbls.add(target.dataset);
      for (const l of visibleLinks) {
        const a = colKey(l.from.dataset, l.from.column);
        const b = colKey(l.to.dataset, l.to.column);
        if (cols.has(a) || cols.has(b)) {
          lks.add(l.id);
          cols.add(a).add(b);
          tbls.add(l.from.dataset).add(l.to.dataset);
        }
      }
    } else if (target.type === "link") {
      const l = visibleLinks.find((x) => x.id === target.id);
      if (l) {
        lks.add(l.id);
        cols.add(colKey(l.from.dataset, l.from.column)).add(colKey(l.to.dataset, l.to.column));
        tbls.add(l.from.dataset).add(l.to.dataset);
      }
    } else if (target.type === "table") {
      tbls.add(target.id);
      for (const l of visibleLinks) {
        if (l.from.dataset === target.id || l.to.dataset === target.id) {
          lks.add(l.id);
          tbls.add(l.from.dataset).add(l.to.dataset);
          cols.add(colKey(l.from.dataset, l.from.column)).add(colKey(l.to.dataset, l.to.column));
        }
      }
      for (const r of visibleRels) if (r.from === target.id || r.to === target.id) tbls.add(r.from).add(r.to);
    } else if (target.type === "relationship") {
      const r = visibleRels.find((x) => x.id === target.id);
      if (r) {
        tbls.add(r.from).add(r.to);
        for (const l of visibleLinks) if (l.relationship === r.id) {
          lks.add(l.id);
          cols.add(colKey(l.from.dataset, l.from.column)).add(colKey(l.to.dataset, l.to.column));
        }
      }
    }
    return { cols, tbls, lks };
  }, [selection, hovered, visibleLinks, visibleRels]);

  const searchHit = useCallback((text: string) => filters.search.trim().length > 1 && text.toLowerCase().includes(filters.search.trim().toLowerCase()), [filters.search]);

  // columns to draw per table, handle sides per column, and node heights
  const drawn = useMemo(() => {
    const perTable = new Map<string, { columns: SchemaColumn[]; hidden: number; sides: Map<string, Set<"l" | "r">> }>();
    if (!map) return perTable;
    const needed = new Map<string, Set<string>>();
    const sides = new Map<string, Map<string, Set<"l" | "r">>>();
    const addSide = (ds: string, col: string, side: "l" | "r") => {
      if (!sides.has(ds)) sides.set(ds, new Map());
      const m = sides.get(ds)!;
      if (!m.has(col)) m.set(col, new Set());
      m.get(col)!.add(side);
    };
    for (const l of visibleLinks) {
      for (const end of [l.from, l.to]) {
        if (!needed.has(end.dataset)) needed.set(end.dataset, new Set());
        needed.get(end.dataset)!.add(end.column);
      }
      const [lf, lt] = [layerOf(l.from.dataset), layerOf(l.to.dataset)];
      if (lf < lt) { addSide(l.from.dataset, l.from.column, "r"); addSide(l.to.dataset, l.to.column, "l"); }
      else if (lf > lt) { addSide(l.from.dataset, l.from.column, "l"); addSide(l.to.dataset, l.to.column, "r"); }
      else { addSide(l.from.dataset, l.from.column, "r"); addSide(l.to.dataset, l.to.column, "r"); }
    }
    for (const t of map.tables) {
      const show = filters.allColumns || expanded.has(t.id);
      const need = needed.get(t.id) || new Set();
      const columns = t.columns.filter((c) => show || need.has(c.name) || c.is_key || searchHit(c.name));
      perTable.set(t.id, { columns, hidden: t.columns.length - columns.length, sides: sides.get(t.id) || new Map() });
    }
    return perTable;
  }, [map, visibleLinks, filters.allColumns, expanded, layerOf, searchHit]);

  const onColumn = useCallback((ds: string, col: string) => setSelection((s) => (s?.type === "column" && s.dataset === ds && s.column === col ? null : { type: "column", dataset: ds, column: col })), []);
  const onTable = useCallback((id: string) => setSelection((s) => (s?.type === "table" && s.id === id ? null : { type: "table", id })), []);
  const onToggle = useCallback((id: string) => setExpanded((e) => { const n = new Set(e); n.has(id) ? n.delete(id) : n.add(id); return n; }), []);

  // layered layout: x from the backend layer, y stacked with real node heights; one block per connected component
  const { nodes, edges } = useMemo(() => {
    if (!map) return { nodes: [] as Node[], edges: [] as Edge[] };
    const heightOf = (t: SchemaTable) => {
      if (mode === "datasets") return HEADER_H;
      const d = drawn.get(t.id)!;
      return HEADER_H + d.columns.length * ROW_H + (d.hidden > 0 || expanded.has(t.id) ? FOOTER_H : 0) + 4;
    };
    const nodes: Node[] = [];
    const blocks = new Map<string, SchemaTable[]>();
    for (const t of map.tables) {
      const k = t.unconnected ? "unconnected" : `c${t.component}`;
      blocks.set(k, [...(blocks.get(k) || []), t]);
    }
    const layerW = TABLE_W + (mode === "datasets" ? LAYER_GAP + 80 : LAYER_GAP);
    let y0 = 0;
    const multi = map.stats.components > 1 || map.stats.unconnected.length > 0;
    for (const [key, ts] of blocks) {
      if (multi) {
        nodes.push({ id: `label:${key}`, type: "label", position: { x: 0, y: y0 }, data: { label: key === "unconnected" ? "Datasets with no relationship found" : `Connected group ${Number(key.slice(1)) + 1}` }, draggable: false, selectable: false });
        y0 += 34;
      }
      let blockH = 0;
      if (key === "unconnected") {
        let x = 0;
        let rowH = 0;
        ts.forEach((t, i) => {
          if (i > 0 && i % 4 === 0) { y0 += rowH + ROW_GAP; x = 0; rowH = 0; }
          nodes.push(tableNode(t, x, y0));
          rowH = Math.max(rowH, heightOf(t));
          x += TABLE_W + 60;
        });
        blockH = rowH;
      } else {
        const layers = new Map<number, SchemaTable[]>();
        ts.forEach((t) => layers.set(t.layer, [...(layers.get(t.layer) || []), t]));
        for (const [layer, lts] of layers) {
          let y = y0;
          lts.sort((a, b) => a.order - b.order).forEach((t) => {
            nodes.push(tableNode(t, layer * layerW, y));
            y += heightOf(t) + ROW_GAP;
          });
          blockH = Math.max(blockH, y - y0);
        }
      }
      y0 += blockH + BLOCK_GAP;
    }

    function tableNode(t: SchemaTable, x: number, y: number): Node {
      const d = drawn.get(t.id)!;
      const f = focus ? (focus.tbls.has(t.id) ? "focus" : "dim") : "none";
      const highlightCols = new Set<string>();
      if (focus) for (const k of focus.cols) { const [ds, col] = k.split("\u0000"); if (ds === t.id) highlightCols.add(col); }
      return {
        id: t.id, type: "table", position: { x, y },
        data: { table: t, columns: d.columns, hidden: d.hidden, mode, focus: f, highlightCols, handleSides: d.sides, onColumn, onTable, onToggle, expanded: expanded.has(t.id), searchHit } satisfies TableData,
      };
    }

    const edges: Edge[] = [];
    // permanent edge labels only while they cannot pile up; otherwise on hover / selection (all listed in the side panel)
    const quietLabels = visibleRels.length > 6;
    if (mode === "columns") {
      const labelled = new Set<string>();
      for (const l of visibleLinks) {
        const [lf, lt] = [layerOf(l.from.dataset), layerOf(l.to.dataset)];
        const [sSide, tSide] = lf < lt ? (["r", "l"] as const) : lf > lt ? (["l", "r"] as const) : (["r", "r"] as const);
        const color = linkColor(l);
        const active = focus?.lks.has(l.id);
        const dim = focus && !active;
        const isKey = l.kind === "join_key";
        const firstOfRel = isKey && l.relationship && !labelled.has(l.relationship);
        if (firstOfRel) labelled.add(l.relationship!);
        const rel = l.relationship ? map.relationships.find((r) => r.id === l.relationship) : undefined;
        edges.push({
          id: l.id,
          source: l.from.dataset,
          target: l.to.dataset,
          sourceHandle: handleId(sSide, "s", l.from.column),
          targetHandle: handleId(tSide, "t", l.to.column),
          type: sSide === tSide ? "smoothstep" : "default",
          zIndex: active ? 10 : isKey ? 5 : 1,
          style: {
            stroke: color,
            strokeWidth: active ? 3.5 : isKey ? 2.4 : 1.4,
            strokeDasharray: l.kind === "candidate" || l.kind === "rejected" ? "1 5" : !isKey ? "5 4" : l.join_kind === "aggregate_lookup" ? "8 4" : undefined,
            opacity: dim ? 0.12 : l.kind === "candidate" || l.kind === "rejected" ? 0.6 : 1,
          },
          markerEnd: isKey && (l.join_kind === "lookup" || l.join_kind === "aggregate_lookup") ? { type: MarkerType.ArrowClosed, color, width: 16, height: 16 } : undefined,
          label: active || (firstOfRel && !focus && !quietLabels) ? (isKey ? `${rel?.cardinality ?? ""} · ${l.confidence.toFixed(2)}${l.in_merge_tree ? " · in merge" : ""}` : `${l.kind === "attribute" ? "same data" : l.kind} · ${l.confidence.toFixed(2)}`) : undefined,
          labelStyle: { fill: "#e8ebf8", fontSize: 10.5 },
          labelBgStyle: { fill: "#111831", fillOpacity: 0.92 },
          labelBgPadding: [4, 2] as [number, number],
          labelBgBorderRadius: 4,
          data: l,
        });
      }
    } else {
      for (const r of visibleRels) {
        const [lf, lt] = [layerOf(r.from), layerOf(r.to)];
        const [sSide, tSide] = lf < lt ? (["r", "l"] as const) : lf > lt ? (["l", "r"] as const) : (["r", "r"] as const);
        const color = FAMILY_COLOR[r.join_kind] || NEUTRAL;
        const active = focus?.tbls.has(r.from) && focus?.tbls.has(r.to) && (selection?.type !== "relationship" || selection.id === r.id);
        const keys = r.key_pairs.length === 1 ? (r.key_pairs[0].from === r.key_pairs[0].to ? r.key_pairs[0].from : `${r.key_pairs[0].from} → ${r.key_pairs[0].to}`) : `${r.key_pairs.length} matched columns`;
        edges.push({
          id: r.id, source: r.from, target: r.to, sourceHandle: `${sSide}s:__table`, targetHandle: `${tSide}t:__table`,
          type: sSide === tSide ? "smoothstep" : "default",
          style: { stroke: color, strokeWidth: r.in_merge_tree ? 3 : 2, strokeDasharray: r.join_kind === "aggregate_lookup" || r.join_kind === "entity_resolution" ? "8 4" : undefined, opacity: focus && !active ? 0.15 : 1 },
          markerEnd: r.join_kind.includes("lookup") ? { type: MarkerType.ArrowClosed, color } : undefined,
          label: !quietLabels || (focus && active) ? `${keys} · ${r.cardinality} · ${r.confidence.toFixed(2)}` : undefined,
          labelStyle: { fill: "#e8ebf8", fontSize: 10.5 },
          labelBgStyle: { fill: "#111831", fillOpacity: 0.92 },
          labelBgPadding: [4, 2] as [number, number],
          labelBgBorderRadius: 4,
          data: r,
        });
      }
    }
    return { nodes, edges };
  }, [map, mode, drawn, visibleLinks, visibleRels, focus, expanded, layerOf, onColumn, onTable, onToggle, searchHit, selection]);

  // evidence for a selected column link
  useEffect(() => {
    setExplanation(null);
    if (selection?.type !== "link" || !map) return;
    const l = map.links.find((x) => x.id === selection.id);
    if (!l) return;
    api
      .get<MatchExplanation>(`/integration/explain/match?left=${encodeURIComponent(`${l.from.dataset}.${l.from.column}`)}&right=${encodeURIComponent(`${l.to.dataset}.${l.to.column}`)}`)
      .then(setExplanation)
      .catch(() => setExplanation(null));
  }, [selection, map]);

  if (!canDiscover) return <Empty title="Load at least two datasets" hint="The schema map shows which column of which dataset connects to which column of another, and how the datasets are joined." />;

  const set = <K extends keyof Filters>(k: K, v: Filters[K]) => setFilters((f) => ({ ...f, [k]: v }));
  const counts = map?.stats.links_by_kind || {};

  return (
    <div className={fullscreen ? "fixed inset-0 z-50 flex bg-ink-950" : "flex h-full min-h-0"} data-testid="graph-root">
      <div className="flex min-w-0 flex-1 flex-col">
        {/* toolbar: view, filters, search */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-ink-800 px-3 py-1.5 text-[11.5px]">
          <div className="flex overflow-hidden rounded-lg border border-ink-700">
            {(["columns", "datasets"] as const).map((m) => (
              <button key={m} onClick={() => { setMode(m); setSelection(null); }} className={`px-3 py-1 text-xs ${mode === m ? "bg-accent-500 text-white" : "bg-ink-800 text-ink-300"}`}>
                {m === "columns" ? "Column links" : "Dataset overview"}
              </button>
            ))}
          </div>
          {mode === "columns" && (
            <>
              <label className="flex items-center gap-1 text-ink-300"><input type="checkbox" checked={filters.joinKeys} onChange={(e) => set("joinKeys", e.target.checked)} /> join keys ({counts.join_key || 0})</label>
              <label className="flex items-center gap-1 text-ink-300"><input type="checkbox" checked={filters.attributes} onChange={(e) => set("attributes", e.target.checked)} /> shared attributes ({counts.attribute || 0})</label>
              <label className="flex items-center gap-1 text-ink-300"><input type="checkbox" checked={filters.candidates} onChange={(e) => set("candidates", e.target.checked)} /> candidates ({(counts.candidate || 0) + (counts.rejected || 0)})</label>
              <label className="flex items-center gap-1 text-ink-300"><input type="checkbox" checked={filters.allColumns} onChange={(e) => set("allColumns", e.target.checked)} /> all columns</label>
            </>
          )}
          <label className="flex items-center gap-1 text-ink-300"><input type="checkbox" checked={filters.mergeOnly} onChange={(e) => set("mergeOnly", e.target.checked)} /> used in merge plan</label>
          <label className="flex items-center gap-1.5 text-ink-300">
            min confidence
            <input type="range" min={0} max={1} step={0.05} value={filters.minConfidence} onChange={(e) => set("minConfidence", Number(e.target.value))} className="w-20" />
            <span className="w-7 font-mono text-ink-100">{filters.minConfidence.toFixed(2)}</span>
          </label>
          <input value={filters.search} onChange={(e) => set("search", e.target.value)} placeholder="find column or dataset" className="w-44 rounded-md border border-ink-700 bg-ink-900 px-2 py-0.5 text-ink-100 placeholder:text-ink-500" />
          {loading && <Spinner />}
          <button onClick={() => setFullscreen((f) => !f)} className="ml-auto rounded-md border border-ink-700 bg-ink-800 px-2 py-0.5 text-xs text-ink-200 hover:bg-ink-700" title={fullscreen ? "Exit full screen (Esc)" : "Full screen"}>
            {fullscreen ? "✕ exit full screen" : "⤢ full screen"}
          </button>
        </div>
        {/* legend */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-ink-800 px-3 py-1 text-[10.5px] text-ink-400" data-testid="graph-legend">
          <LegendLine color={FAMILY_COLOR.lookup} label="reference: many rows → one key (arrow points to the key)" />
          <LegendLine color={FAMILY_COLOR.lookup} dash="8 4" label="aggregated lookup" />
          <LegendLine color={FAMILY_COLOR.entity_key_merge} label="same entity via shared key" />
          <LegendLine color={FAMILY_COLOR.entity_resolution} label="same entity, matched probabilistically" />
          {mode === "columns" && <LegendLine color={NEUTRAL} dash="5 4" label="same information, not joined on" />}
          {mode === "columns" && <span>🔑 key · ↗ reference · ⇄ linked · click a column, table or line</span>}
        </div>
        <div className="relative min-h-0 flex-1" onMouseLeave={() => setHovered(null)}>
          <ErrorNote error={error} />
          <ReactFlow
            key={`${mode}-${map?.tables.length ?? 0}`}
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.1 }}
            minZoom={0.1}
            nodesConnectable={false}
            onEdgeClick={(_, e) => setSelection(mode === "columns" ? { type: "link", id: e.id } : { type: "relationship", id: e.id })}
            onNodeMouseEnter={(_, n) => n.type === "table" && setHovered(n.id)}
            onNodeMouseLeave={() => setHovered(null)}
            onPaneClick={() => setSelection(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#26305a" gap={22} />
            {(map?.tables.length ?? 0) > 4 && (
              <MiniMap pannable zoomable position="top-right" nodeColor={(n) => (n.type === "table" ? "#3a4577" : "transparent")} maskColor="rgba(11,16,32,0.7)" style={{ background: "#111831", width: 160, height: 100 }} />
            )}
            <Controls showInteractive={false} position="bottom-left" />
          </ReactFlow>
        </div>
      </div>

      <aside className="w-[380px] shrink-0 overflow-y-auto border-l border-ink-800 p-4">
        {!map ? (
          error ? <ErrorNote error={error} /> : <Spinner />
        ) : (
          <SidePanel map={map} selection={selection} setSelection={setSelection} explanation={explanation} visibleLinks={visibleLinks} visibleRels={visibleRels} sendToChat={sendToChat} />
        )}
      </aside>
    </div>
  );
}

function LegendLine({ color, label, dash }: { color: string; label: string; dash?: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="26" height="6" aria-hidden><line x1="0" y1="3" x2="26" y2="3" stroke={color} strokeWidth="2.5" strokeDasharray={dash} /></svg>
      {label}
    </span>
  );
}

/* ------------------------------------------------------------------ side panel: details + an accessible list of every connection */
function SidePanel({
  map, selection, setSelection, explanation, visibleLinks, visibleRels, sendToChat,
}: {
  map: SchemaMap;
  selection: Selection;
  setSelection: (s: Selection) => void;
  explanation: MatchExplanation | null;
  visibleLinks: SchemaLink[];
  visibleRels: SchemaRelationship[];
  sendToChat: ((text: string) => void) | null;
}) {
  const table = (id: string) => map.tables.find((t) => t.id === id);
  const linkRow = (l: SchemaLink, from?: string) => (
    <button key={l.id} onClick={() => setSelection({ type: "link", id: l.id })} className="flex w-full items-start gap-2 rounded-md px-1.5 py-1 text-left hover:bg-ink-800">
      <span className="mt-1.5 inline-block h-0.5 w-3 shrink-0" style={{ background: linkColor(l), height: l.kind === "join_key" ? 3 : 1 }} />
      <span className="min-w-0 flex-1">
        <span className="block break-all font-mono text-[11.5px] leading-snug">
          {from !== l.from.dataset && <span className="text-ink-400">{l.from.dataset}.</span>}{l.from.column}
          <span className="text-ink-400"> → </span>
          {from !== l.to.dataset && <span className="text-ink-400">{l.to.dataset}.</span>}{l.to.column}
        </span>
        <span className="block text-[10.5px] text-ink-400">
          {l.kind === "join_key" ? "join key" : l.kind === "attribute" ? "same information" : l.kind} · confidence {l.confidence.toFixed(2)}{l.in_merge_tree ? " · used in merge" : ""}
        </span>
      </span>
    </button>
  );

  if (selection?.type === "column") {
    const t = table(selection.dataset);
    const c = t?.columns.find((x) => x.name === selection.column);
    const links = visibleLinks.filter((l) => (l.from.dataset === selection.dataset && l.from.column === selection.column) || (l.to.dataset === selection.dataset && l.to.column === selection.column));
    return (
      <div className="space-y-3 text-sm" data-testid="graph-side-panel">
        <div className="text-xs uppercase tracking-wide text-ink-400">Column</div>
        <div className="font-mono text-[13px]">{selection.dataset}.<span className="text-white">{selection.column}</span></div>
        {c && (
          <div className="flex flex-wrap gap-1.5">
            <Badge tone="accent">{ROLE_ICON[c.role].icon} {c.role}</Badge>
            <Badge>{c.semantic_type.toLowerCase()}</Badge>
            <Badge>{c.data_type.toLowerCase()}</Badge>
            <Badge>{pct(c.uniqueness)} unique</Badge>
            <Badge>{pct(c.null_pct)} null</Badge>
          </div>
        )}
        {c?.description && <div className="text-[12px] text-ink-300">Glossary: {c.description.replace(/_/g, " ")}</div>}
        <div className="text-[12.5px] text-ink-300">{c ? ROLE_ICON[c.role].title : ""}</div>
        <div>
          <div className="mb-1 text-xs text-ink-400">{links.length ? `Connected to ${links.length} column${links.length === 1 ? "" : "s"}` : "No connections with the current filters"}</div>
          {links.map((l) => linkRow(l))}
        </div>
      </div>
    );
  }

  if (selection?.type === "link") {
    const l = map.links.find((x) => x.id === selection.id);
    const r = l?.relationship ? map.relationships.find((x) => x.id === l.relationship) : undefined;
    return (
      <div className="space-y-3 text-sm" data-testid="graph-side-panel">
        {l && (
          <div className="rounded-lg border border-ink-700 p-2.5 text-[12.5px]">
            <div className="font-mono">{l.from.dataset}.<b>{l.from.column}</b></div>
            <div className="text-ink-400">{l.kind === "join_key" ? "joins to" : l.kind === "attribute" ? "holds the same information as" : "may correspond to"}</div>
            <div className="font-mono">{l.to.dataset}.<b>{l.to.column}</b></div>
            {r && <div className="mt-1.5 text-[11.5px] text-ink-300">Part of the {KIND_LABEL[r.join_kind] || r.join_kind} relationship {r.from} → {r.to} ({r.cardinality}){l.in_merge_tree ? ", used by the merge plan" : ""}.</div>}
            {l.row_agreement != null && (
              <div className="mt-1 text-[11.5px] text-ink-300">Checked on {l.rows_compared?.toLocaleString()} key-joined rows: values agree in {Math.round(l.row_agreement * 100)}%.</div>
            )}
            {l.rejection_reason && l.kind !== "join_key" && l.kind !== "attribute" && <div className="mt-1 text-[11.5px] text-amber-200">Not accepted: {l.rejection_reason}</div>}
          </div>
        )}
        {explanation ? <EvidencePanel explanation={explanation} /> : <Spinner />}
      </div>
    );
  }

  const relFocus = selection?.type === "relationship" ? visibleRels.filter((r) => r.id === selection.id) : selection?.type === "table" ? visibleRels.filter((r) => r.from === selection.id || r.to === selection.id) : visibleRels;
  const t = selection?.type === "table" ? table(selection.id) : undefined;
  const unconnected = map.stats.unconnected;

  return (
    <div className="space-y-4 text-sm" data-testid="graph-side-panel">
      {t ? (
        <div>
          <div className="text-xs uppercase tracking-wide text-ink-400">Dataset</div>
          <div className="font-semibold">{t.label}</div>
          <div className="text-[12px] text-ink-400">{fmt(t.rows ?? 0)} rows · {t.column_count} columns · {t.connected_columns} connected · keys: {t.columns.filter((c) => c.is_key).map((c) => c.name).join(", ") || "none detected"}</div>
        </div>
      ) : (
        <div>
          <div className="text-xs uppercase tracking-wide text-ink-400">How the datasets connect</div>
          <div className="text-[12px] text-ink-400">
            {map.stats.tables} datasets · {map.stats.relationships} relationships · {(map.stats.links_by_kind.join_key || 0) + (map.stats.links_by_kind.attribute || 0)} column links
            {unconnected.length > 0 && ` · ${unconnected.length} not connected`}
          </div>
        </div>
      )}
      {relFocus.length === 0 && <div className="text-[12px] text-ink-400">No relationships with the current filters.</div>}
      {relFocus.map((r) => {
        const links = visibleLinks.filter((l) => l.relationship === r.id);
        return (
          <div key={r.id} className="rounded-lg border border-ink-700 p-2.5">
            <button className="w-full text-left" onClick={() => setSelection({ type: "relationship", id: r.id })}>
              <div className="flex items-center gap-2 text-[12.5px]">
                <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: FAMILY_COLOR[r.join_kind] || NEUTRAL }} />
                <span className="truncate font-semibold">{r.from} → {r.to}</span>
              </div>
              <div className="mt-0.5 flex flex-wrap gap-1.5 text-[10.5px]">
                <Badge tone="accent">{KIND_LABEL[r.join_kind] || r.join_kind}</Badge>
                <Badge>{r.cardinality}</Badge>
                <Badge tone={r.confidence >= 0.9 ? "good" : r.confidence >= 0.7 ? "warn" : "bad"}>confidence {r.confidence.toFixed(2)}</Badge>
                {r.in_merge_tree && <Badge tone="good">used in merge</Badge>}
              </div>
            </button>
            <div className="mt-1.5">{links.map((l) => linkRow(l))}</div>
            {selection?.type === "relationship" && <p className="mt-1.5 text-[12px] leading-relaxed text-ink-300">{r.explanation}</p>}
            {selection?.type === "relationship" && sendToChat && (
              <Button className="mt-2 w-full" onClick={() => sendToChat(`Explain why ${r.from} and ${r.to} were linked`)}>Ask for a full explanation</Button>
            )}
          </div>
        );
      })}
      {!t && unconnected.length > 0 && (
        <div>
          <div className="mb-1 text-xs text-ink-400">No relationship found for</div>
          {unconnected.map((u) => <div key={u} className="font-mono text-[11.5px] text-ink-300">{u}</div>)}
        </div>
      )}
    </div>
  );
}
