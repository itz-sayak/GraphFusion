"use client";

import { ReactNode } from "react";

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" | "accent"; title?: string }) {
  const tones: Record<string, string> = {
    neutral: "bg-ink-700 text-ink-300",
    good: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-400/30",
    warn: "bg-amber-500/15 text-amber-300 ring-1 ring-amber-400/30",
    bad: "bg-red-500/15 text-red-300 ring-1 ring-red-400/30",
    accent: "bg-accent-500/15 text-accent-300 ring-1 ring-accent-400/30",
  };
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function bandTone(band?: string | null): "good" | "warn" | "bad" | "neutral" {
  if (band === "HIGH") return "good";
  if (band === "MEDIUM") return "warn";
  if (band === "LOW") return "bad";
  return "neutral";
}

export function confidenceTone(c: number): "good" | "warn" | "bad" {
  return c >= 0.9 ? "good" : c >= 0.7 ? "warn" : "bad";
}

export function Button({
  children,
  onClick,
  variant = "secondary",
  disabled,
  title,
  className = "",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
  title?: string;
  className?: string;
  type?: "button" | "submit";
}) {
  const variants: Record<string, string> = {
    primary: "bg-accent-500 hover:bg-accent-400 text-white",
    secondary: "bg-ink-700 hover:bg-ink-600 text-ink-100",
    ghost: "hover:bg-ink-800 text-ink-300",
    danger: "bg-red-500/20 hover:bg-red-500/30 text-red-200",
  };
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return <span className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-400 border-t-transparent ${className}`} />;
}

export function Empty({ title, hint, action }: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="flex h-full min-h-[160px] flex-col items-center justify-center gap-2 p-6 text-center">
      <div className="text-sm font-medium text-ink-300">{title}</div>
      {hint && <div className="max-w-md text-xs text-ink-400">{hint}</div>}
      {action}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="m-3 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">{error}</div>;
}

export function ConfidenceBar({ value, className = "" }: { value: number; className?: string }) {
  const color = value >= 0.9 ? "bg-emerald-400" : value >= 0.7 ? "bg-amber-400" : "bg-red-400";
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-full bg-ink-700 ${className}`}>
      <div className={`h-full ${color}`} style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

export function fmt(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function cellText(v: unknown): string {
  if (v === null || v === undefined) return "∅";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
