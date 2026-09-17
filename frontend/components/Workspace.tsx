"use client";

import { MouseEvent as ReactMouseEvent, useCallback, useRef, useState } from "react";
import { TabId, useStore } from "@/lib/store";
import ConflictsView from "./tabs/ConflictsView";
import GraphView from "./tabs/GraphView";
import MappingsView from "./tabs/MappingsView";
import OutputView from "./tabs/OutputView";
import PlanView from "./tabs/PlanView";
import ProvenanceView from "./tabs/ProvenanceView";
import ReviewView from "./tabs/ReviewView";
import SchemaView from "./tabs/SchemaView";

const TABS: { id: TabId; label: string }[] = [
  { id: "graph", label: "Graph" },
  { id: "schema", label: "Schema" },
  { id: "mappings", label: "Mappings" },
  { id: "plan", label: "Merge plan" },
  { id: "conflicts", label: "Conflicts" },
  { id: "review", label: "Review" },
  { id: "output", label: "Output" },
  { id: "provenance", label: "Provenance" },
];

export default function Workspace() {
  const { tab, setTab, session } = useStore();
  const [height, setHeight] = useState(48);
  const container = useRef<HTMLDivElement>(null);
  const activeMerge = session?.merges.find((m) => m.status === "active" && m === session.merges.filter((x) => x.status === "active").slice(-1)[0]);

  const startDrag = useCallback((e: ReactMouseEvent) => {
    e.preventDefault();
    const parent = container.current?.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const move = (ev: MouseEvent) => setHeight(Math.min(85, Math.max(18, ((rect.bottom - ev.clientY) / rect.height) * 100)));
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }, []);

  return (
    <div ref={container} className="flex min-h-0 flex-col border-t border-ink-800 bg-ink-950" style={{ height: `${height}%` }}>
      <div onMouseDown={startDrag} className="h-1.5 cursor-row-resize bg-ink-900 hover:bg-accent-500/40" title="Drag to resize" />
      <div className="flex items-center gap-1 border-b border-ink-800 px-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`relative px-3 py-2 text-[13px] transition ${tab === t.id ? "text-white" : "text-ink-400 hover:text-ink-100"}`}
          >
            {t.label}
            {t.id === "conflicts" && activeMerge && activeMerge.conflicts > 0 && <span className="ml-1 rounded bg-amber-500/20 px-1 text-[10px] text-amber-200">{activeMerge.conflicts}</span>}
            {tab === t.id && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded bg-accent-500" />}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1">
        {tab === "graph" && <GraphView />}
        {tab === "schema" && <SchemaView />}
        {tab === "mappings" && <MappingsView />}
        {tab === "plan" && <PlanView />}
        {tab === "conflicts" && <ConflictsView />}
        {tab === "review" && <ReviewView />}
        {tab === "output" && <OutputView />}
        {tab === "provenance" && <ProvenanceView />}
      </div>
    </div>
  );
}
