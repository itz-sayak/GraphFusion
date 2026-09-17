"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Badge, Empty, ErrorNote, Spinner } from "../ui";

interface Hop {
  operation: string;
  dataset?: string;
  column?: string;
  join_keys?: string;
  role?: string | null;
  confidence?: number;
  strategy?: string;
  members?: string[];
  aggregate?: string;
  method?: string;
  transformation?: string | null;
}
interface Lineage {
  semantic_type: string;
  data_type: string;
  sources: { dataset: string; column: string; transformation: string | null }[];
  path: Hop[];
  merge_operation: string;
}
interface ProvenanceResponse {
  merge_id: string;
  columns: Record<string, Lineage>;
  row_lineage_columns: string[];
}

const OP_TONE: Record<string, "good" | "accent" | "warn" | "neutral"> = { select: "neutral", lookup: "good", fuse: "accent", aggregate: "warn", entity_resolution: "accent" };

export default function ProvenanceView() {
  const { session, version } = useStore();
  const [data, setData] = useState<ProvenanceResponse | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hasMerge = session?.merges.some((m) => m.status === "active");

  useEffect(() => {
    if (!hasMerge) return;
    Promise.all([api.get<ProvenanceResponse>("/integration/provenance"), api.get<string>("/integration/lineage-report")])
      .then(([p, r]) => {
        setData(p);
        setReport(r);
        setSelected((s) => s && p.columns[s] ? s : Object.keys(p.columns).find((c) => !c.startsWith("_")) || null);
      })
      .catch((e) => setError(e.message));
  }, [version, hasMerge]);

  if (!hasMerge) return <Empty title="No merge yet" hint="Every unified column is traced to its source columns, transformations and merge operations." />;
  if (!data) return error ? <ErrorNote error={error} /> : <div className="p-6"><Spinner /></div>;
  const lin = selected ? data.columns[selected] : null;

  return (
    <div className="flex h-full min-h-0">
      <div className="w-64 shrink-0 overflow-y-auto border-r border-ink-800 py-2">
        {Object.entries(data.columns).map(([col, l]) => (
          <button
            key={col}
            onClick={() => setSelected(col)}
            className={`flex w-full items-center justify-between px-3 py-1 text-left text-[12px] ${selected === col ? "bg-accent-500/15 text-white" : "text-ink-300 hover:bg-ink-900"}`}
          >
            <span className="truncate font-mono">{col}</span>
            <span className="text-[10px] text-ink-400">{l.merge_operation}</span>
          </button>
        ))}
      </div>
      <div className="min-w-0 flex-1 overflow-y-auto p-5">
        {lin && selected && (
          <div className="space-y-4">
            <div>
              <div className="font-mono text-lg text-white">unified.{selected}</div>
              <div className="mt-1 flex gap-1.5">
                <Badge>{lin.semantic_type}</Badge>
                <Badge>{lin.data_type}</Badge>
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-xs uppercase tracking-wide text-ink-400">Ultimate sources</div>
              {lin.sources.map((s, i) => (
                <div key={i} className="font-mono text-[13px]">
                  ← {s.dataset}.{s.column} {s.transformation && <span className="text-accent-300">[{s.transformation}]</span>}
                </div>
              ))}
            </div>
            <div>
              <div className="mb-1.5 text-xs uppercase tracking-wide text-ink-400">Derivation path</div>
              <ol className="space-y-2">
                {lin.path.map((h, i) => (
                  <li key={i} className="flex gap-3">
                    <div className="mt-0.5 h-5 w-5 shrink-0 rounded-full bg-ink-700 text-center text-[11px] leading-5">{i + 1}</div>
                    <div className="text-[12.5px]">
                      <Badge tone={OP_TONE[h.operation] || "neutral"}>{h.operation}</Badge>{" "}
                      {h.dataset && <span className="font-mono">{h.dataset}{h.column ? `.${h.column}` : ""}</span>}
                      {h.role && <span className="ml-1 text-accent-300">role {h.role}</span>}
                      {h.join_keys && <div className="mt-0.5 font-mono text-[11px] text-ink-400">on {h.join_keys}</div>}
                      {h.confidence !== undefined && <div className="text-[11px] text-ink-400">relationship confidence {h.confidence.toFixed(2)}</div>}
                      {h.strategy && <div className="text-[11px] text-ink-400">conflict strategy {h.strategy}{h.members ? ` over ${h.members.join(", ")}` : ""}</div>}
                      {h.aggregate && <div className="text-[11px] text-ink-400">aggregate {h.aggregate}</div>}
                      {h.method && <div className="text-[11px] text-ink-400">method {h.method}</div>}
                    </div>
                  </li>
                ))}
              </ol>
            </div>
            <div className="text-[12px] text-ink-400">
              Row-level lineage is stored in the dataset itself: <span className="font-mono">{data.row_lineage_columns.join(", ")}</span>. Cell-level provenance of fused values is in <span className="font-mono">cell_provenance.parquet</span>.
            </div>
          </div>
        )}
        {report && (
          <details className="mt-6">
            <summary className="cursor-pointer text-xs uppercase tracking-wide text-ink-400">Full lineage report</summary>
            <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-ink-800 bg-ink-900 p-3 font-mono text-[11px] text-ink-300">{report}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
