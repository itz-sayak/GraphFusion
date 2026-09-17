"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { Conflict, HistoryResponse } from "@/lib/types";
import { Badge, Empty, ErrorNote, Spinner, cellText } from "../ui";

interface ConflictsResponse {
  total: number;
  matching: number;
  by_attribute: Record<string, number>;
  strategy: string;
  conflicts: Conflict[];
}

export default function ConflictsView() {
  const { session, version } = useStore();
  const [data, setData] = useState<ConflictsResponse | null>(null);
  const [attribute, setAttribute] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ entity: string; data: HistoryResponse | null } | null>(null);

  const openHistory = (entity: string) => {
    if (history?.entity === entity) return setHistory(null);
    setHistory({ entity, data: null });
    api.get<HistoryResponse>(`/integration/history?entity_id=${encodeURIComponent(entity)}`)
      .then((data) => setHistory({ entity, data }))
      .catch((e) => setError(e.message));
  };
  const hasMerge = session?.merges.some((m) => m.status === "active");

  useEffect(() => {
    if (!hasMerge) return;
    setError(null);
    api
      .get<ConflictsResponse>(`/integration/conflicts?limit=500${attribute ? `&attribute=${encodeURIComponent(attribute)}` : ""}`)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [version, attribute, hasMerge]);

  if (!hasMerge) return <Empty title="No merge yet" hint="Conflicts appear when the records of one entity disagree on an attribute. Nothing is overwritten silently." />;
  if (!data) return error ? <ErrorNote error={error} /> : <div className="p-6"><Spinner /></div>;
  if (!data.total) return <Empty title="No conflicts" hint="Every fused attribute agreed across sources after normalisation." />;

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">{data.total} conflicts</span>
        <Badge tone="accent">strategy: {data.strategy}</Badge>
        <button onClick={() => setAttribute(null)} className={`rounded-md px-2 py-0.5 text-xs ${!attribute ? "bg-accent-500 text-white" : "bg-ink-800 text-ink-300"}`}>all</button>
        {Object.entries(data.by_attribute).map(([a, n]) => (
          <button key={a} onClick={() => setAttribute(a)} className={`rounded-md px-2 py-0.5 text-xs ${attribute === a ? "bg-accent-500 text-white" : "bg-ink-800 text-ink-300"}`}>
            {a} · {n}
          </button>
        ))}
      </div>
      <div className="space-y-2">
        {data.conflicts.map((c, i) => (
          <div key={c.entity_id + c.attribute + i} className="rounded-xl border border-ink-800 p-3">
            <div className="flex flex-wrap items-center gap-2 text-[13px]">
              <button onClick={() => openHistory(c.entity_id)} title="Show how this entity's attributes changed over time" className="font-mono text-ink-400 underline decoration-dotted hover:text-ink-100">{c.entity_id}</button>
              <span className="font-semibold">{c.attribute}</span>
              <span className="text-ink-400">→</span>
              <span className="rounded bg-emerald-500/15 px-1.5 font-mono text-emerald-200">{cellText(c.resolved_value)}</span>
              <Badge tone={c.status === "manual_review" ? "warn" : "neutral"}>{c.status}</Badge>
              <span className="text-[12px] text-ink-400">{c.reason}</span>
            </div>
            <table className="mt-2 w-full text-[12px]">
              <tbody>
                {c.candidates.map((x, j) => (
                  <tr key={j} className={x.selected ? "text-emerald-200" : "text-ink-300"}>
                    <td className="w-6 py-0.5">{x.selected ? "✔" : ""}</td>
                    <td className="py-0.5 font-mono">{cellText(x.value)}</td>
                    <td className="py-0.5 font-mono text-[11px] text-ink-400">{x.source_dataset}.{x.source_column} · row {x.source_row}</td>
                    <td className="py-0.5 text-[11px] text-ink-400">confidence {x.confidence.toFixed(2)}</td>
                    <td className="py-0.5 text-[11px] text-ink-400">{x.record_timestamp ? `updated ${x.record_timestamp.slice(0, 16)}` : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {history?.entity === c.entity_id && <HistoryTimeline data={history.data} attribute={c.attribute} />}
          </div>
        ))}
      </div>
    </div>
  );
}

function HistoryTimeline({ data, attribute }: { data: HistoryResponse | null; attribute: string }) {
  if (!data) return <div className="mt-2"><Spinner /></div>;
  const rows = data.rows.filter((r) => r.valid_from);
  if (!rows.length) return <div className="mt-2 text-[12px] text-ink-400" data-testid="history-timeline">No dated changes recorded for this entity.</div>;
  const byAttr: Record<string, typeof rows> = {};
  rows.forEach((r) => (byAttr[r.attribute] = [...(byAttr[r.attribute] || []), r]));
  return (
    <div className="mt-2 rounded-lg border border-ink-800 p-2" data-testid="history-timeline">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-400">Attribute history (valid time)</div>
      {Object.entries(byAttr).map(([attr, list]) => (
        <div key={attr} className={`mb-1 flex flex-wrap items-center gap-1 text-[12px] ${attr === attribute ? "" : "opacity-70"}`}>
          <span className="w-28 text-ink-400">{attr}</span>
          {list.map((h, i) => (
            <span key={i} className="flex items-center gap-1">
              {i > 0 && <span className="text-ink-500">→</span>}
              <span className={`rounded px-1.5 py-0.5 font-mono ${h.is_current ? "bg-emerald-500/15 text-emerald-200" : "bg-ink-800 text-ink-200"}`} title={`${h.source_dataset}.${h.source_column} row ${h.source_row} · ${h.timestamp_kind} timestamp`}>
                {cellText(h.value)}
              </span>
              <span className="text-[11px] text-ink-400">{h.valid_from?.slice(0, 10)}{h.is_current ? " – now" : ` – ${h.valid_to?.slice(0, 10)}`}{h.timestamp_kind === "creation" ? " (created)" : ""}</span>
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}
