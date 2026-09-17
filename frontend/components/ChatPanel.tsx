"use client";

import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "@/lib/api";
import { TabId, useStore } from "@/lib/store";
import type { ChatMessage, ChatResponse } from "@/lib/types";
import { Badge, Button, Spinner } from "./ui";

const CARD_TAB: Record<string, TabId> = {
  discovery: "graph",
  graph: "graph",
  mappings: "mappings",
  explanation: "mappings",
  plan: "plan",
  merge: "output",
  conflicts: "conflicts",
  provenance: "provenance",
  profile: "schema",
  export: "output",
  quality: "output",
  stats: "output",
  route: "graph",
  table: "output",
};

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi — I integrate datasets that were never designed to fit together.\n\nUpload files on the left (or load the **sample** or **NYC** demo), then ask me to *find relationships*, *show the merge plan*, or simply *merge them*. I explain every decision with evidence.",
};

export default function ChatPanel() {
  const { session, bump, setTab, registerChat } = useStore();
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>(["Load the sample datasets", "Find relationships between the datasets"]);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || sending) return;
      if (/^load (the )?sample/i.test(message)) {
        await api.post("/demo/sample");
        bump();
      } else if (/^load (the )?nyc/i.test(message)) {
        await api.post("/demo/nyc");
        bump();
      }
      const user: ChatMessage = { id: crypto.randomUUID(), role: "user", content: message };
      const pending: ChatMessage = { id: crypto.randomUUID(), role: "assistant", content: "", pending: true };
      setMessages((m) => [...m, user, pending]);
      setInput("");
      setSending(true);
      try {
        const res = await api.post<ChatResponse>("/chat", { message: /^load (the )?(sample|nyc)/i.test(message) ? "Find relationships between the datasets" : message, session_id: session?.session_id });
        setMessages((m) => m.map((x) => (x.id === pending.id ? { id: x.id, role: "assistant", content: res.reply, tool_calls: res.tool_calls, provider: res.provider, elapsed_ms: res.elapsed_ms } : x)));
        setSuggestions(res.suggestions);
        const lastCard = res.cards[res.cards.length - 1];
        if (lastCard && CARD_TAB[lastCard.type]) setTab(CARD_TAB[lastCard.type]);
        bump();
      } catch (e) {
        setMessages((m) => m.map((x) => (x.id === pending.id ? { ...x, pending: false, content: `⚠️ ${e instanceof Error ? e.message : String(e)}` } : x)));
      } finally {
        setSending(false);
      }
    },
    [sending, session?.session_id, bump, setTab],
  );

  useEffect(() => {
    registerChat((t: string) => void send(t));
  }, [registerChat, send]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(input);
    }
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <div ref={scroller} className="flex-1 space-y-4 overflow-y-auto px-6 py-5">
        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[88%] rounded-2xl px-4 py-2.5 text-[13.5px] leading-relaxed ${
                m.role === "user" ? "bg-accent-500 text-white" : "border border-ink-800 bg-ink-900"
              }`}
            >
              {m.pending ? (
                <div className="flex items-center gap-2 text-ink-300">
                  <Spinner /> Working on it…
                </div>
              ) : m.role === "user" ? (
                <div className="whitespace-pre-wrap">{m.content}</div>
              ) : (
                <div className="prose-chat">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                </div>
              )}
              {m.tool_calls && m.tool_calls.length > 0 && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-ink-800 pt-2">
                  {m.tool_calls.map((t, i) => (
                    <Badge key={i} tone={t.ok ? "accent" : "bad"} title={t.error || JSON.stringify(t.arguments)}>
                      {t.tool}
                      {Object.keys(t.arguments || {}).length ? `(${Object.entries(t.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})` : "()"}
                    </Badge>
                  ))}
                  {m.provider && (
                    <span className="ml-auto text-[10px] text-ink-400">
                      {m.provider.fallback ? "rule-based fallback" : m.provider.provider} · {((m.elapsed_ms || 0) / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-ink-800 px-6 pb-4 pt-3">
        <div className="mb-2 flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button
              key={s}
              disabled={sending}
              onClick={() => void send(s)}
              className="rounded-full border border-ink-700 px-2.5 py-1 text-[11.5px] text-ink-300 transition hover:border-accent-400 hover:text-white disabled:opacity-40"
            >
              {s}
            </button>
          ))}
        </div>
        <form onSubmit={onSubmit} className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={2}
            placeholder='Ask e.g. "Merge them, but only matches above 95%" or "Why were these datasets linked?"'
            className="min-h-[44px] flex-1 resize-none rounded-xl border border-ink-700 bg-ink-900 px-3 py-2 text-sm outline-none placeholder:text-ink-400 focus:border-accent-500"
          />
          <Button type="submit" variant="primary" disabled={sending || !input.trim()} className="h-[44px] px-4">
            {sending ? <Spinner /> : "Send"}
          </Button>
        </form>
      </div>
    </section>
  );
}
