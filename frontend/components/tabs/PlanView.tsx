"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { MergePlan } from "@/lib/types";
import { Badge, Button, Empty, ErrorNote, Spinner } from "../ui";

const OP_TONE: Record<string, "accent" | "good" | "warn" | "neutral"> = {
  transform: "neutral",
  flag_duplicates: "warn",
  resolve_entities: "accent",
  fuse_entities: "accent",
  lookup: "good",
  aggregate_lookup: "warn",
  reverse_aggregate: "warn",
  preserve_provenance: "neutral",
  validate: "neutral",
  materialize: "neutral",
};

export default function PlanView() {
  const { session, version, bump, setTab } = useStore();
  const [plan, setPlan] = useState<MergePlan | null>(null);
  const [busy, setBusy] = useState<"plan" | "execute" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!session?.plan_id) {
      setPlan(null);
      return;
    }
    api.get<MergePlan>("/integration/plan").then(setPlan).catch(() => setPlan(null));
  }, [version, session?.plan_id]);

  const generate = async () => {
    setBusy("plan");
    setError(null);
    try {
      setPlan(await api.post<MergePlan>("/integration/plan", {}));
      bump();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const execute = async () => {
    setBusy("execute");
    setError(null);
    try {
      await api.post("/integration/execute", { plan_id: plan?.plan_id });
      bump();
      setTab("output");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  if ((session?.datasets.length || 0) < 2) return <Empty title="Load at least two datasets" />;
  if (!plan)
    return (
      <Empty
        title="No merge plan yet"
        hint="The plan is derived from the maximum spanning tree of the integration graph and lists merge order, join keys, join types, transformations, conflict rules and thresholds."
        action={<Button variant="primary" onClick={() => void generate()} disabled={!!busy}>{busy ? <Spinner /> : "Generate plan"}</Button>}
      />
    );

  return (
    <div className="h-full overflow-auto p-4">
      <ErrorNote error={error} />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-ink-400">{plan.plan_id}</span>
        <Badge tone="accent">mode {plan.mode}</Badge>
        <Badge>conflicts: {plan.conflict_strategy}</Badge>
        <Badge>auto-merge ≥ {plan.thresholds.auto_merge}</Badge>
        <Badge>review ≥ {plan.thresholds.review}</Badge>
        <Badge>relationships ≥ {plan.thresholds.min_edge}</Badge>
        <div className="ml-auto flex gap-2">
          <Button onClick={() => void generate()} disabled={!!busy}>{busy === "plan" ? <Spinner /> : "Regenerate"}</Button>
          <Button variant="primary" onClick={() => void execute()} disabled={!!busy}>{busy === "execute" ? <><Spinner /> Executing…</> : "Execute merge"}</Button>
        </div>
      </div>

      <div className="mb-4 rounded-xl border border-ink-800 bg-ink-900 p-3 text-sm">
        <div><span className="text-ink-400">Root / grain:</span> <b>{plan.root_dataset}</b> — {plan.grain}</div>
        {plan.integration_tree.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[12.5px]">
            <span className="text-ink-400">Integration tree:</span>
            {plan.integration_tree.map((e) => (
              <span key={e.parent + e.child} className="rounded-md bg-ink-800 px-2 py-0.5 font-mono">
                {e.parent} → {e.child} <span className="text-accent-300">[{e.join_kind}, {e.confidence.toFixed(2)}]</span>
              </span>
            ))}
          </div>
        )}
        {plan.routes.length > 0 && (
          <div className="mt-2 text-[12.5px] text-ink-300">
            <span className="text-ink-400">Dijkstra routes from root:</span>{" "}
            {plan.routes.map((r) => `${r.path.join(" → ")} (reliability ${r.reliability.toFixed(2)})`).join(";  ")}
          </div>
        )}
      </div>

      <ol className="space-y-2">
        {plan.steps.map((s) => (
          <li key={s.step} className="flex gap-3 rounded-xl border border-ink-800 p-3">
            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink-700 text-xs font-semibold">{s.step}</div>
            <div className="min-w-0 flex-1">
              <div className="mb-1 flex flex-wrap items-center gap-1.5">
                <Badge tone={OP_TONE[s.operation] || "neutral"}>{s.operation.replace(/_/g, " ")}</Badge>
                {s.join_type && <Badge>{s.join_type} join</Badge>}
                {s.confidence !== null && <Badge tone={s.confidence >= 0.9 ? "good" : "warn"}>confidence {s.confidence.toFixed(2)}</Badge>}
                {s.requires_review && <Badge tone="warn">review recommended</Badge>}
              </div>
              <div className="text-[13px] leading-relaxed text-ink-100">{s.description}</div>
            </div>
          </li>
        ))}
      </ol>

      {plan.excluded_relationships.length > 0 && (
        <div className="mt-5">
          <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Relationships not used</div>
          {plan.excluded_relationships.map((e) => (
            <div key={e.left + e.right} className="text-[12.5px] text-ink-300">
              <span className="font-mono">{e.left} ↔ {e.right}</span> ({e.confidence.toFixed(2)}): {e.reason}
            </div>
          ))}
        </div>
      )}
      {plan.warnings.length > 0 && (
        <div className="mt-4 space-y-1">
          {plan.warnings.map((w) => <div key={w} className="rounded-md bg-amber-500/10 px-2 py-1 text-[12px] text-amber-200">{w}</div>)}
        </div>
      )}

      <div className="mt-5">
        <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Unified schema ({plan.canonical_schema.length} columns, dynamically derived)</div>
        <div className="overflow-auto rounded-lg border border-ink-800">
          <table className="w-full text-[12px]">
            <thead className="bg-ink-900 text-left text-ink-400">
              <tr><th className="px-2 py-1">column</th><th className="px-2 py-1">operation</th><th className="px-2 py-1">semantic type</th><th className="px-2 py-1">sources</th><th className="px-2 py-1">transformation</th></tr>
            </thead>
            <tbody>
              {plan.canonical_schema.map((c) => (
                <tr key={c.name} className="border-t border-ink-900">
                  <td className="px-2 py-1 font-mono text-white">{c.name}</td>
                  <td className="px-2 py-1 text-ink-300">{c.role}</td>
                  <td className="px-2 py-1 text-ink-300">{c.semantic_type}</td>
                  <td className="px-2 py-1 font-mono text-[11px] text-ink-300">{c.sources.map((s) => `${s.dataset_id}.${s.column}`).join(", ")}</td>
                  <td className="px-2 py-1 text-ink-400">{c.transformation || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
