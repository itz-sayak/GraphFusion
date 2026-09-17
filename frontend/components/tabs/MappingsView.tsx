"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { MatchExplanation } from "@/lib/types";
import EvidencePanel from "../EvidencePanel";
import { Badge, Button, ConfidenceBar, Empty, ErrorNote, Spinner, bandTone } from "../ui";

interface MappingExport {
  mappings: MatchExplanation[];
  dataset_relationships: { left_dataset: string; right_dataset: string; confidence: number; band: string; join_kind: string; explanation: string; keys: string[] }[];
  flooding: { iterations: number; converged: boolean } | null;
  candidate_strategy: string | null;
}

export default function MappingsView() {
  const { session, version, bump } = useStore();
  const [data, setData] = useState<MappingExport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRejected, setShowRejected] = useState(false);
  const [selected, setSelected] = useState<MatchExplanation | null>(null);

  useEffect(() => {
    if ((session?.datasets.length || 0) < 2) return;
    api.get<MappingExport>("/integration/mappings").then(setData).catch((e) => setError(e.message));
  }, [version, session?.datasets.length]);

  const [retrain, setRetrain] = useState<string | null>(null);
  const scorer = session?.preferences?.scorer || "weighted";

  const setScorer = async (value: string) => {
    await api.post("/integration/preferences", { scorer: value });
    bump();
  };

  const retrainMatcher = async () => {
    setRetrain("Retraining…");
    try {
      const r = await api.post<{ labels_used: string[]; metadata: { user_labels?: number }; note: string | null }>("/integration/matcher/retrain", {});
      setRetrain(`Retrained with ${r.labels_used.length} decision${r.labels_used.length === 1 ? "" : "s"}.${r.note ? " " + r.note : ""}`);
      bump();
    } catch (e) {
      setRetrain(e instanceof Error ? e.message : String(e));
    }
  };

  const decide = async (m: MatchExplanation, decision: "approved" | "rejected") => {
    await api.post("/integration/mappings/decision", { left: m.left, right: m.right, decision });
    bump();
  };

  if ((session?.datasets.length || 0) < 2) return <Empty title="Load at least two datasets" hint="Column correspondences are computed across datasets." />;
  if (!data) return error ? <ErrorNote error={error} /> : <div className="p-6"><Spinner /></div>;
  const rows = data.mappings.filter((m) => showRejected || m.accepted);

  return (
    <div className="flex h-full min-h-0">
      <div className="min-w-0 flex-1 overflow-auto p-4">
        <div className="mb-3 flex items-center gap-3 text-xs text-ink-400">
          <span>{data.mappings.filter((m) => m.accepted).length} accepted of {data.mappings.length} scored correspondences</span>
          <span>candidates: {data.candidate_strategy}</span>
          {data.flooding && <span>similarity flooding: {data.flooding.iterations} iterations{data.flooding.converged ? " (converged)" : ""}</span>}
          <label className="ml-auto flex items-center gap-1.5">
            scorer
            <select className="rounded-md border border-ink-700 bg-ink-800 px-1.5 py-0.5 text-xs" value={scorer} onChange={(e) => void setScorer(e.target.value)}>
              <option value="weighted">weighted</option>
              <option value="learned">learned</option>
              <option value="blend">blend</option>
            </select>
          </label>
          <Button variant="ghost" className="px-2 py-0.5 text-[11px]" onClick={() => void retrainMatcher()} title="Refit the learned matcher using your approve/reject decisions">
            Retrain matcher from my decisions
          </Button>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} /> show rejected
          </label>
        </div>
        {retrain && <div className="mb-2 rounded-md bg-accent-500/10 px-2 py-1 text-[11.5px] text-accent-300">{retrain}</div>}
        <table className="w-full text-[12.5px]">
          <thead className="text-left text-ink-400">
            <tr>
              <th className="px-2 py-1.5 font-medium">left</th>
              <th className="px-2 py-1.5 font-medium">right</th>
              <th className="w-40 px-2 py-1.5 font-medium">confidence</th>
              <th className="px-2 py-1.5 font-medium">values via</th>
              <th className="px-2 py-1.5 font-medium">graph Δ</th>
              <th className="px-2 py-1.5 font-medium">status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.left + m.right} onClick={() => setSelected(m)} className={`cursor-pointer border-t border-ink-900 hover:bg-ink-900 ${selected?.left === m.left && selected?.right === m.right ? "bg-ink-900" : ""}`}>
                <td className="px-2 py-1.5 font-mono">{m.left.replace("::", ".")}</td>
                <td className="px-2 py-1.5 font-mono">{m.right.replace("::", ".")}</td>
                <td className="px-2 py-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-9 font-mono">{m.confidence.toFixed(2)}</span>
                    <ConfidenceBar value={m.confidence} />
                  </div>
                </td>
                <td className="px-2 py-1.5 font-mono text-[11px] text-ink-300">{m.normalizer}</td>
                <td className={`px-2 py-1.5 font-mono text-[11px] ${m.graph.adjustment >= 0 ? "text-emerald-300" : "text-red-300"}`}>{m.graph.adjustment >= 0 ? "+" : ""}{m.graph.adjustment.toFixed(3)}</td>
                <td className="px-2 py-1.5">
                  <Badge tone={m.accepted ? bandTone(m.band) : "bad"}>{m.accepted ? m.band : "rejected"}</Badge>
                </td>
                <td className="px-2 py-1.5 text-right" onClick={(e) => e.stopPropagation()}>
                  {m.accepted ? (
                    <Button variant="ghost" className="px-2 py-0.5 text-[11px]" onClick={() => void decide(m, "rejected")}>reject</Button>
                  ) : (
                    <Button variant="ghost" className="px-2 py-0.5 text-[11px]" onClick={() => void decide(m, "approved")}>approve</Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="mt-6 text-xs uppercase tracking-wide text-ink-400">Dataset relationships</div>
        <div className="mt-2 space-y-2">
          {data.dataset_relationships.map((r) => (
            <div key={r.left_dataset + r.right_dataset} className="rounded-lg border border-ink-800 p-3">
              <div className="flex items-center gap-2 text-sm">
                <span className="font-semibold">{r.left_dataset} ↔ {r.right_dataset}</span>
                <Badge tone="accent">{r.join_kind.replace(/_/g, " ")}</Badge>
                <Badge tone={bandTone(r.band)}>{r.confidence.toFixed(2)}</Badge>
              </div>
              <div className="mt-1 text-[12.5px] text-ink-300">{r.explanation}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="w-[360px] shrink-0 overflow-y-auto border-l border-ink-800 p-4">
        {selected ? <EvidencePanel explanation={selected} /> : <Empty title="Select a mapping" hint="Every score is decomposed into its signals, weights and graph adjustments." />}
      </div>
    </div>
  );
}
