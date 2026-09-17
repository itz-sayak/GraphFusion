"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { ReviewPair, ReviewQueue } from "@/lib/types";
import { Badge, Button, ConfidenceBar, Empty, ErrorNote, Spinner, cellText } from "../ui";

const LEVEL_TONE: Record<string, "good" | "warn" | "bad" | "neutral"> = { exact: "good", high: "good", medium: "warn", different: "bad", null: "neutral" };

function pairKey(p: ReviewPair): string {
  return `${p.left.dataset}:${p.left.row}|${p.right.dataset}:${p.right.row}`;
}

export default function ReviewView() {
  const { session, version, bump } = useStore();
  const [data, setData] = useState<ReviewQueue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [labelled, setLabelled] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const hasMerge = session?.merges.some((m) => m.status === "active");

  const load = useCallback(() => {
    setError(null);
    api.get<ReviewQueue>("/integration/entities/review?limit=20").then((d) => {
      setData(d);
      setLabelled({});
    }).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (hasMerge) load();
  }, [version, hasMerge, load]);

  const label = async (p: ReviewPair, decision: "match" | "non_match") => {
    try {
      await api.post("/integration/entities/label", { left: { dataset: p.left.dataset, row: p.left.row }, right: { dataset: p.right.dataset, row: p.right.row }, decision, sampling: p.sampling });
      setLabelled((s) => ({ ...s, [pairKey(p)]: decision }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const remerge = async () => {
    setBusy(true);
    try {
      await api.post("/integration/execute", {});
      bump();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!hasMerge) return <Empty title="No merge yet" hint="After a merge with entity resolution, the record pairs the model is least sure about appear here for you to label." />;
  if (!data) return error ? <ErrorNote error={error} /> : <div className="p-6"><Spinner /></div>;
  const nLabelled = Object.keys(labelled).length;
  if (!data.pairs.length) return <Empty title="Nothing to review" hint="This merge has no uncertain record pairs (or no entity resolution step)." />;

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">{data.pairs.length} record pairs to review</span>
        <Badge tone="accent">{data.labels + nLabelled} label{data.labels + nLabelled === 1 ? "" : "s"} in session</Badge>
        <span className="text-[12px] text-ink-400">Uncertain pairs sit closest to the merge threshold. Random pairs are unbiased samples that also recalibrate match probabilities.</span>
        <div className="ml-auto flex gap-2">
          <Button variant="ghost" onClick={load}>Refresh</Button>
          <Button variant="primary" disabled={busy || data.labels + nLabelled === 0} onClick={remerge}>{busy ? "Merging…" : "Re-run merge with labels"}</Button>
        </div>
      </div>
      <ErrorNote error={error} />
      <div className="space-y-2">
        {data.pairs.map((p) => {
          const done = labelled[pairKey(p)];
          const fields = Array.from(new Set([...Object.keys(p.left.values), ...Object.keys(p.right.values)]));
          return (
            <div key={pairKey(p)} className={`rounded-xl border p-3 ${done ? "border-ink-800 opacity-60" : "border-ink-700"}`} data-testid="review-pair">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-[13px]">
                <span className="w-28 text-ink-300">match probability</span>
                <ConfidenceBar value={p.probability} className="w-32" />
                <span className="font-mono">{p.probability.toFixed(3)}</span>
                <Badge tone={p.sampling === "random" ? "neutral" : "warn"}>{p.sampling === "random" ? "random sample" : "uncertain"}</Badge>
                <Badge tone="neutral">{p.same_entity_now ? "currently merged" : "currently separate"}</Badge>
                <span className="text-[11px] text-ink-400">{p.group_id}</span>
                <div className="ml-auto flex gap-2">
                  {done ? (
                    <Badge tone={done === "match" ? "good" : "bad"}>{done === "match" ? "labelled: match" : "labelled: not a match"}</Badge>
                  ) : (
                    <>
                      <Button onClick={() => label(p, "match")}>Same entity</Button>
                      <Button variant="danger" onClick={() => label(p, "non_match")}>Different</Button>
                    </>
                  )}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className="text-left text-ink-400">
                      <th className="w-28 py-0.5 font-normal">field</th>
                      <th className="py-0.5 font-normal">{p.left.source} · row {p.left.row}</th>
                      <th className="py-0.5 font-normal">{p.right.source} · row {p.right.row}</th>
                      <th className="w-24 py-0.5 font-normal">comparison</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fields.map((f) => (
                      <tr key={f} className="text-ink-200">
                        <td className="py-0.5 text-ink-400">{f}</td>
                        <td className="py-0.5 font-mono">{cellText(p.left.values[f])}</td>
                        <td className="py-0.5 font-mono">{cellText(p.right.values[f])}</td>
                        <td className="py-0.5">{p.field_levels[f] ? <Badge tone={LEVEL_TONE[p.field_levels[f]] || "neutral"}>{p.field_levels[f]}</Badge> : null}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
