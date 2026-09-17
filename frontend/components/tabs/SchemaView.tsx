"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import type { DatasetProfile } from "@/lib/types";
import { Badge, Empty, ErrorNote, Spinner, cellText, fmt } from "../ui";

interface Preview {
  columns: string[];
  rows: unknown[][];
  total_rows: number;
}

export default function SchemaView() {
  const { session, selectedDataset, selectDataset, version } = useStore();
  const datasetId = selectedDataset || session?.datasets[0]?.dataset_id || null;
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;
    setProfile(null);
    setError(null);
    Promise.all([api.get<DatasetProfile>(`/datasets/${datasetId}/profile`), api.get<Preview>(`/datasets/${datasetId}/preview?limit=12`)])
      .then(([p, v]) => {
        setProfile(p);
        setPreview(v);
      })
      .catch((e) => setError(e.message));
  }, [datasetId, version]);

  if (!session?.datasets.length) return <Empty title="No datasets loaded" hint="Upload files or load a demo to inspect schemas." />;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-ink-800 px-4 py-2">
        {session.datasets.map((d) => (
          <button
            key={d.dataset_id}
            onClick={() => selectDataset(d.dataset_id)}
            className={`rounded-lg px-2.5 py-1 text-xs ${d.dataset_id === datasetId ? "bg-accent-500 text-white" : "bg-ink-800 text-ink-300 hover:bg-ink-700"}`}
          >
            {d.source_name}
          </button>
        ))}
      </div>
      <ErrorNote error={error} />
      {!profile ? (
        <div className="p-6">
          <Spinner />
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <div className="mb-3 flex flex-wrap gap-2 text-xs">
            <Badge>{fmt(profile.row_count)} rows</Badge>
            <Badge>{profile.column_count} columns</Badge>
            <Badge tone={profile.duplicate_rows ? "warn" : "good"}>{profile.duplicate_rows} duplicate rows</Badge>
            <Badge>profiled in {fmt(profile.elapsed_ms)} ms</Badge>
          </div>
          <table className="w-full text-left text-[12px]">
            <thead className="sticky top-0 bg-ink-950 text-ink-400">
              <tr>
                {["column", "type", "semantic type", "nulls", "unique", "cardinality", "min", "max", "mean", "patterns", "samples"].map((h) => (
                  <th key={h} className="border-b border-ink-800 px-2 py-1.5 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name} className="border-b border-ink-900 hover:bg-ink-900">
                  <td className="px-2 py-1.5 font-mono text-white">{c.name}</td>
                  <td className="px-2 py-1.5 text-ink-300">{c.data_type}</td>
                  <td className="px-2 py-1.5">
                    <Badge tone={c.semantic_confidence >= 0.7 ? "accent" : "neutral"} title={JSON.stringify(c.semantic_evidence)}>
                      {c.semantic_type} {c.semantic_confidence.toFixed(2)}
                    </Badge>
                  </td>
                  <td className={`px-2 py-1.5 ${c.null_pct > 0.2 ? "text-amber-300" : "text-ink-300"}`}>{(c.null_pct * 100).toFixed(1)}%</td>
                  <td className="px-2 py-1.5 text-ink-300">{fmt(c.unique_count)}</td>
                  <td className="px-2 py-1.5 text-ink-300">{c.cardinality}</td>
                  <td className="max-w-[120px] truncate px-2 py-1.5 text-ink-300">{cellText(c.min)}</td>
                  <td className="max-w-[120px] truncate px-2 py-1.5 text-ink-300">{cellText(c.max)}</td>
                  <td className="px-2 py-1.5 text-ink-300">{c.mean !== null ? fmt(c.mean, 2) : "—"}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px] text-ink-400">{Object.keys(c.pattern_histogram).slice(0, 2).join("  ")}</td>
                  <td className="max-w-[240px] truncate px-2 py-1.5 text-ink-400">{c.sample_values.slice(0, 3).map(cellText).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {preview && (
            <div className="mt-5">
              <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Preview (first {preview.rows.length} of {fmt(preview.total_rows)} rows)</div>
              <div className="overflow-auto rounded-lg border border-ink-800">
                <table className="text-[11.5px]">
                  <thead className="bg-ink-900 text-ink-400">
                    <tr>{preview.columns.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1 text-left font-medium">{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((r, i) => (
                      <tr key={i} className="border-t border-ink-900">
                        {r.map((v, j) => <td key={j} className={`whitespace-nowrap px-2 py-1 ${v === null ? "text-ink-600" : "text-ink-300"}`}>{cellText(v)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
