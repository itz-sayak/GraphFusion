"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Badge, Button, Empty, ErrorNote, Spinner, cellText, fmt } from "../ui";

interface OutputPreview {
  merge_id: string;
  columns: string[];
  rows: unknown[][];
  total_rows: number;
}
interface Quality {
  quality_report: { text: string; integration_coverage: number; conflicts: number; output_rows: number; resolved_entities: number };
  validation: { passed: boolean; errors: number; warnings: number; checks: { check: string; passed: boolean; severity: string; details: string }[] };
  stats: { match_percentage: number };
}

export default function OutputView() {
  const { session, version, bump } = useStore();
  const [preview, setPreview] = useState<OutputPreview | null>(null);
  const [quality, setQuality] = useState<Quality | null>(null);
  const [files, setFiles] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hideLineage, setHideLineage] = useState(true);
  const hasMerge = session?.merges.some((m) => m.status === "active");

  useEffect(() => {
    if (!hasMerge) return;
    setError(null);
    Promise.all([
      api.get<OutputPreview>(`/integration/output?limit=50&offset=${offset}`),
      api.get<Quality>("/integration/quality"),
      api.get<{ files: Record<string, number> }>("/integration/export"),
    ])
      .then(([p, q, f]) => {
        setPreview(p);
        setQuality(q);
        setFiles(f.files);
      })
      .catch((e) => setError(e.message));
  }, [version, offset, hasMerge]);

  if (!hasMerge) return <Empty title="No unified dataset yet" hint='Ask "Merge them" in the chat or execute the plan.' />;
  if (!preview || !quality) return error ? <ErrorNote error={error} /> : <div className="p-6"><Spinner /></div>;

  const visible = preview.columns.map((c, i) => [c, i] as const).filter(([c]) => !hideLineage || !(c.startsWith("_src_") || c.startsWith("_match_")));

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-ink-800 px-4 py-2 text-xs">
          <span className="font-mono text-ink-400">{preview.merge_id}</span>
          <Badge>{fmt(preview.total_rows)} rows</Badge>
          <Badge>{preview.columns.length} columns</Badge>
          <Badge tone={quality.validation.passed ? "good" : "bad"}>validation {quality.validation.passed ? "passed" : "failed"}</Badge>
          <Badge tone="accent">coverage {(quality.quality_report.integration_coverage * 100).toFixed(1)}%</Badge>
          <label className="ml-auto flex items-center gap-1.5 text-ink-400">
            <input type="checkbox" checked={hideLineage} onChange={(e) => setHideLineage(e.target.checked)} /> hide lineage columns
          </label>
          <Button variant="ghost" className="px-2 py-0.5 text-xs" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>prev</Button>
          <span className="text-ink-400">{fmt(offset + 1)}–{fmt(Math.min(offset + 50, preview.total_rows))}</span>
          <Button variant="ghost" className="px-2 py-0.5 text-xs" disabled={offset + 50 >= preview.total_rows} onClick={() => setOffset(offset + 50)}>next</Button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="text-[11.5px]">
            <thead className="sticky top-0 bg-ink-900 text-ink-400">
              <tr>{visible.map(([c]) => <th key={c} className={`whitespace-nowrap px-2 py-1.5 text-left font-medium ${c.startsWith("_") ? "text-ink-600" : ""}`}>{c}</th>)}</tr>
            </thead>
            <tbody>
              {preview.rows.map((r, i) => (
                <tr key={i} className="border-t border-ink-900 hover:bg-ink-900">
                  {visible.map(([c, j]) => (
                    <td key={c} className={`max-w-[260px] truncate whitespace-nowrap px-2 py-1 ${r[j] === null ? "text-ink-600" : c.startsWith("_") ? "text-ink-400" : "text-ink-100"}`}>{cellText(r[j])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="w-[380px] shrink-0 space-y-4 overflow-y-auto border-l border-ink-800 p-4">
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Export</div>
          <div className="space-y-1">
            {Object.entries(files).map(([name, size]) => (
              <a key={name} href={api.downloadUrl(name)} className="flex items-center justify-between rounded-md px-2 py-1 text-[12.5px] hover:bg-ink-800" download>
                <span className="font-mono text-accent-300">{name}</span>
                <span className="text-[11px] text-ink-400">{size > 1e6 ? `${(size / 1e6).toFixed(1)} MB` : `${(size / 1e3).toFixed(1)} KB`}</span>
              </a>
            ))}
          </div>
        </div>
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Data quality report</div>
          <pre className="whitespace-pre-wrap rounded-lg border border-ink-800 bg-ink-900 p-3 font-mono text-[11px] leading-relaxed text-ink-300">{quality.quality_report.text}</pre>
        </div>
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">Validation checks</div>
          <div className="space-y-1">
            {quality.validation.checks.map((c) => (
              <div key={c.check} className="text-[11.5px]" title={c.details}>
                <span className={c.passed ? "text-emerald-300" : c.severity === "warning" ? "text-amber-300" : "text-red-300"}>{c.passed ? "✔" : "✖"}</span>{" "}
                <span className="font-mono text-ink-300">{c.check}</span>
                <div className="ml-4 text-[11px] text-ink-400">{c.details}</div>
              </div>
            ))}
          </div>
        </div>
        <Button variant="danger" className="w-full" onClick={() => void api.post("/integration/undo").then(bump)}>
          Undo this merge
        </Button>
      </div>
    </div>
  );
}
