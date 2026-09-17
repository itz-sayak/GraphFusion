"use client";

import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ApiError, api, setSessionId } from "./api";
import type { SessionInfo } from "./types";

export type TabId = "graph" | "schema" | "mappings" | "plan" | "conflicts" | "review" | "output" | "provenance";

interface Store {
  session: SessionInfo | null;
  version: number;
  refresh: () => Promise<void>;
  bump: () => void;
  tab: TabId;
  setTab: (t: TabId) => void;
  selectedDataset: string | null;
  selectDataset: (id: string | null) => void;
  backendError: string | null;
  notice: string | null;
  authRequired: boolean;
  clearNotice: () => void;
  resetSession: () => Promise<void>;
  sendToChat: ((text: string) => void) | null;
  registerChat: (fn: (text: string) => void) => void;
}

const Ctx = createContext<Store | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [version, setVersion] = useState(0);
  const [tab, setTab] = useState<TabId>("graph");
  const [selectedDataset, selectDataset] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const clearNotice = useCallback(() => setNotice(null), []);
  const [sendToChat, setSendToChat] = useState<((text: string) => void) | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await api.get<SessionInfo>("/sessions/current");
      setSession((prev) => {
        // same session id but the backend has no data for it: its workspace was deleted or could not be restored
        if (prev && prev.session_id === s.session_id && prev.datasets.length > 0 && s.datasets.length === 0 && s.merges.length === 0) {
          setNotice("This session's datasets are no longer on the server (its workspace was removed). Load the data again.");
        }
        return s;
      });
      setBackendError(null);
      setAuthRequired(false);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        setAuthRequired(true);
        setBackendError(null);
      } else {
        setBackendError(e instanceof Error ? e.message : String(e));
      }
    }
  }, []);

  const bump = useCallback(() => {
    setVersion((v) => v + 1);
    void refresh();
  }, [refresh]);

  const resetSession = useCallback(async () => {
    setSessionId(null);
    const s = await api.post<SessionInfo>("/sessions");
    setSessionId(s.session_id);
    setSession(s);
    selectDataset(null);
    setVersion((v) => v + 1);
  }, []);

  const registerChat = useCallback((fn: (text: string) => void) => setSendToChat(() => fn), []);

  useEffect(() => {
    void refresh();
    // re-sync when the tab regains focus, so a restarted backend never leaves a stale list on screen
    const onFocus = () => void refresh();
    const onVisible = () => document.visibilityState === "visible" && void refresh();
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [refresh]);

  const value = useMemo(
    () => ({ session, version, refresh, bump, tab, setTab, selectedDataset, selectDataset, backendError, notice, clearNotice, authRequired, resetSession, sendToChat, registerChat }),
    [session, version, refresh, bump, tab, selectedDataset, backendError, notice, clearNotice, authRequired, resetSession, sendToChat, registerChat],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside provider");
  return s;
}
