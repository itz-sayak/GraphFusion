"use client";

import { DragEvent, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Badge, Button, Spinner, fmt } from "./ui";

const FORMAT_TONE: Record<string, "accent" | "good" | "warn" | "neutral"> = { csv: "good", parquet: "accent", json: "warn", jsonl: "warn", xlsx: "good", rest: "neutral", sqlite: "neutral" };

export default function DatasetSidebar() {
  const { session, bump, selectedDataset, selectDataset, setTab, notice, clearNotice } = useStore();
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const input = useRef<HTMLInputElement>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setBusy(`Ingesting ${files.length} file${files.length > 1 ? "s" : ""}…`);
    setErrors([]);
    try {
      const res = await api.upload<{ datasets: unknown[]; errors: { file: string; error: string }[] }>(files);
      setErrors(res.errors.map((e) => `${e.file}: ${e.error}`));
      bump();
    } catch (e) {
      setErrors([e instanceof Error ? e.message : String(e)]);
    } finally {
      setBusy(null);
    }
  };

  const loadDemo = async (name: "sample" | "nyc") => {
    setBusy(name === "nyc" ? "Loading NYC mobility sources (500K trips)…" : "Loading sample datasets…");
    setErrors([]);
    try {
      await api.post(`/demo/${name}`);
      bump();
    } catch (e) {
      setErrors([e instanceof Error ? e.message : String(e)]);
    } finally {
      setBusy(null);
    }
  };

  const remove = async (id: string) => {
    setErrors([]);
    try {
      await api.del(`/datasets/${id}`);
      if (selectedDataset === id) selectDataset(null);
    } catch (e) {
      const status = e instanceof ApiError ? e.status : 0;
      setErrors([
        status === 404
          ? `${id} is no longer in the server session (the backend was restarted; sessions are kept in memory). The list has been refreshed — load the data again.`
          : `Could not remove ${id}: ${e instanceof Error ? e.message : String(e)}`,
      ]);
    } finally {
      bump(); // always re-sync the list with the backend
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void upload(Array.from(e.dataTransfer.files));
  };

  const datasets = session?.datasets || [];
  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-ink-800 bg-ink-900/40">
      <div className="flex items-center justify-between px-4 pb-2 pt-3">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-ink-400">Datasets</h2>
        <span className="text-[11px] text-ink-400">{datasets.length}</span>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        className={`mx-3 mb-3 cursor-pointer rounded-xl border border-dashed px-3 py-4 text-center text-xs transition ${
          dragging ? "border-accent-400 bg-accent-500/10 text-accent-300" : "border-ink-600 text-ink-400 hover:border-ink-400"
        }`}
      >
        <div className="font-medium text-ink-300">Drop files or click to upload</div>
        <div className="mt-1">CSV · JSON · JSONL · Parquet · XLSX · SQLite</div>
        <input ref={input} type="file" multiple hidden onChange={(e) => void upload(Array.from(e.target.files || []))} />
      </div>

      <div className="mx-3 mb-3 flex gap-2">
        <Button className="flex-1 text-xs" onClick={() => void loadDemo("sample")} disabled={!!busy}>
          Sample data
        </Button>
        <Button className="flex-1 text-xs" onClick={() => void loadDemo("nyc")} disabled={!!busy}>
          NYC demo
        </Button>
      </div>

      {notice && (
        <div className="mx-3 mb-2 rounded-md bg-amber-500/10 px-2 py-1 text-[11px] text-amber-200" role="status">
          {notice}{" "}
          <button className="underline" onClick={clearNotice}>dismiss</button>
        </div>
      )}
      {busy && (
        <div className="mx-3 mb-2 flex items-center gap-2 text-xs text-ink-300">
          <Spinner /> {busy}
        </div>
      )}
      {errors.map((e) => (
        <div key={e} className="mx-3 mb-2 rounded-md bg-red-500/10 px-2 py-1 text-[11px] text-red-200">
          {e}
        </div>
      ))}

      <div className="flex-1 space-y-1.5 overflow-y-auto px-3 pb-3">
        {datasets.map((d) => (
          <div
            key={d.dataset_id}
            onClick={() => {
              selectDataset(d.dataset_id);
              setTab("schema");
            }}
            className={`group cursor-pointer rounded-xl border px-3 py-2 transition ${
              selectedDataset === d.dataset_id ? "border-accent-500/60 bg-accent-500/10" : "border-ink-800 bg-ink-900 hover:border-ink-600"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="truncate text-sm font-medium" title={d.source_name}>
                {d.source_name}
              </div>
              <Badge tone={FORMAT_TONE[d.source_type] || "neutral"}>{d.source_type}</Badge>
            </div>
            <div className="mt-1 flex items-center justify-between text-[11px] text-ink-400">
              <span>
                {fmt(d.row_count)} rows · {d.column_count} cols
              </span>
              <button
                className="invisible text-ink-400 hover:text-red-300 group-hover:visible"
                title="Remove from session (source file is not affected)"
                onClick={(e) => {
                  e.stopPropagation();
                  void remove(d.dataset_id);
                }}
              >
                remove
              </button>
            </div>
          </div>
        ))}
      </div>

      {session && session.merges.length > 0 && (
        <div className="border-t border-ink-800 px-4 py-3">
          <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-400">Merges</h3>
          {session.merges
            .slice()
            .reverse()
            .slice(0, 5)
            .map((m) => (
              <div key={m.merge_id} className="flex items-center justify-between py-0.5 text-[11px]">
                <span className={`font-mono ${m.status === "undone" ? "text-ink-400 line-through" : m.status === "outdated" ? "text-ink-500" : "text-ink-300"}`}
                      title={m.status === "outdated" ? "Datasets changed after this merge; run a new merge" : undefined}>{m.merge_id}</span>
                <span className="text-ink-400">{m.status === "outdated" ? "outdated · " : ""}{fmt(m.rows)} rows</span>
              </div>
            ))}
        </div>
      )}
    </aside>
  );
}
