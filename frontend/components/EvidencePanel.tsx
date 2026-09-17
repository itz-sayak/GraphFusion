"use client";

import type { MatchExplanation } from "@/lib/types";
import { Badge, bandTone } from "./ui";

export default function EvidencePanel({ explanation: e }: { explanation: MatchExplanation }) {
  const max = Math.max(...e.factors.map((f) => f.contribution), 0.0001);
  return (
    <div className="space-y-3 text-sm">
      <div className="text-xs uppercase tracking-wide text-ink-400">Column relationship</div>
      <div className="font-mono text-[12.5px] leading-snug">
        {e.left.replace("::", ".")}
        <br />↔ {e.right.replace("::", ".")}
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Badge tone={bandTone(e.band)}>confidence {e.confidence.toFixed(2)} · {e.band}</Badge>
        <Badge tone={e.accepted ? "good" : "bad"}>{e.accepted ? "accepted" : "rejected"}</Badge>
        <Badge tone="accent">{e.relationship.replace(/_/g, " ")}</Badge>
      </div>
      {e.rejection_reason && <div className="text-xs text-red-200">{e.rejection_reason}</div>}

      <div>
        <div className="mb-1.5 text-xs text-ink-400">Evidence (value · weight)</div>
        <div className="space-y-1.5">
          {e.factors.map((f) => (
            <div key={f.signal}>
              <div className="flex justify-between text-[12px]">
                <span className="text-ink-300">{f.label}</span>
                <span className="font-mono">
                  {f.value.toFixed(2)} <span className="text-ink-400">× {f.weight.toFixed(2)}</span>
                </span>
              </div>
              <div className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-ink-700">
                <div className="h-full bg-accent-400" style={{ width: `${(f.contribution / max) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-ink-700 p-2.5 text-[12px] text-ink-300">
        <div className="mb-1 text-xs text-ink-400">Graph refinement (similarity flooding)</div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11.5px]">
          <span>base score</span>
          <span className="text-right">{e.graph.base_score.toFixed(3)}</span>
          <span>adjustment</span>
          <span className={`text-right ${e.graph.adjustment >= 0 ? "text-emerald-300" : "text-red-300"}`}>{e.graph.adjustment >= 0 ? "+" : ""}{e.graph.adjustment.toFixed(3)}</span>
          <span>2-hop support</span>
          <span className="text-right">{e.graph.transitive_support.toFixed(3)}</span>
          <span>table coherence</span>
          <span className="text-right">{e.graph.structural_support.toFixed(3)}</span>
          <span>exclusivity</span>
          <span className="text-right">{e.graph.exclusivity.toFixed(3)}</span>
        </div>
      </div>
      <div className="text-[12px] text-ink-300">
        Values compared under normaliser <span className="font-mono text-accent-300">{e.normalizer}</span>. Strongest signal: <b>{e.strongest_factor}</b>; weakest: <b>{e.weakest_factor}</b>.
      </div>
    </div>
  );
}
