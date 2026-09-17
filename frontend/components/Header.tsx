"use client";

import { useEffect, useState } from "react";
import { api, setSessionId, setToken } from "@/lib/api";
import { useStore } from "@/lib/store";
import { Badge, Button } from "./ui";

const MODES = ["strict", "balanced", "permissive"];
const STRATEGIES = ["majority_vote", "prefer_source", "prefer_latest", "prefer_non_null", "highest_confidence", "source_accuracy_vote", "keep_all", "manual_review"];

export default function Header() {
  const { session, bump, resetSession, backendError, authRequired, refresh } = useStore();
  const [tokenInput, setTokenInput] = useState("");
  const [provider, setProvider] = useState<{ provider: string; model: string; mock: boolean } | null>(null);

  useEffect(() => {
    api
      .get<{ llm_provider: { provider: string; model: string; mock: boolean } }>("/health")
      .then((h) => setProvider(h.llm_provider))
      .catch(() => setProvider(null));
  }, []);

  const prefs = session?.preferences;
  const update = async (body: Record<string, unknown>) => {
    await api.post("/integration/preferences", body);
    bump();
  };

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-ink-800 bg-ink-900/80 px-4">
      <div className="flex items-center gap-2">
        <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
          <circle cx="7" cy="8" r="4" fill="#5b8cff" />
          <circle cx="25" cy="9" r="4" fill="#34d399" />
          <circle cx="16" cy="25" r="4" fill="#fbbf24" />
          <path d="M7 8 L25 9 L16 25 Z" fill="none" stroke="#8b94c7" strokeWidth="1.6" />
        </svg>
        <div>
          <div className="text-[15px] font-semibold leading-tight">GraphFusion</div>
          <div className="text-[10px] leading-tight text-ink-400">conversational dataset integration</div>
        </div>
      </div>

      <div className="ml-4 flex items-center gap-2 text-xs text-ink-300">
        <label className="flex items-center gap-1.5">
          Mode
          <select
            className="rounded-md border border-ink-700 bg-ink-800 px-2 py-1 text-xs"
            value={prefs?.mode || "balanced"}
            onChange={(e) => void update({ mode: e.target.value })}
            disabled={!session}
          >
            {MODES.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          Conflicts
          <select
            className="rounded-md border border-ink-700 bg-ink-800 px-2 py-1 text-xs"
            value={prefs?.conflict_strategy || "majority_vote"}
            onChange={(e) => void update({ conflict_strategy: e.target.value })}
            disabled={!session}
          >
            {STRATEGIES.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        {prefs?.confidence_threshold ? (
          <Badge tone="accent" title="Minimum confidence set in conversation">
            ≥ {Math.round(prefs.confidence_threshold * 100)}%
            <button className="ml-1 text-ink-300 hover:text-white" onClick={() => void update({ clear_threshold: true })} title="clear">
              ×
            </button>
          </Badge>
        ) : null}
        {prefs?.primary_key ? <Badge tone="accent">key: {prefs.primary_key}</Badge> : null}
        {prefs?.merge_uncertain === false ? <Badge tone="warn">no uncertain merges</Badge> : null}
      </div>

      <div className="ml-auto flex items-center gap-3">
        {authRequired && (
          <form
            className="flex items-center gap-1.5"
            onSubmit={(e) => {
              e.preventDefault();
              setToken(tokenInput.trim() || null);
              setSessionId(null);
              void refresh();
            }}
          >
            <Badge tone="warn">API token required</Badge>
            <input type="password" value={tokenInput} onChange={(e) => setTokenInput(e.target.value)} placeholder="token" aria-label="API token"
              className="w-32 rounded-md border border-ink-700 bg-ink-900 px-2 py-0.5 text-xs text-ink-100" />
            <Button type="submit" variant="primary">Sign in</Button>
          </form>
        )}
        {backendError ? (
          <Badge tone="bad" title={backendError}>
            backend unreachable
          </Badge>
        ) : provider ? (
          <Badge tone={provider.mock ? "neutral" : "good"} title={provider.model}>
            LLM: {provider.provider}
            {provider.mock ? "" : ` · ${provider.model.split("/").pop()}`}
          </Badge>
        ) : null}
        {session && <span className="font-mono text-[11px] text-ink-400">session {session.session_id}</span>}
        <Button variant="ghost" onClick={() => void resetSession()} title="Start a new empty session">
          New session
        </Button>
      </div>
    </header>
  );
}
