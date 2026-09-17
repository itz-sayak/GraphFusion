"""Shared experiment utilities: metrics, timing/memory measurement, tables and charts."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DFG_LOG_LEVEL", "WARNING")

import psutil  # noqa: E402

# Validated categorical palette (fixed order, light surface) — see docs/evaluation.md
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"


def prf(predicted: set, truth: set) -> dict[str, float]:
    tp = len(predicted & truth)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "predicted": len(predicted), "truth": len(truth)}


def best_threshold(scores: dict[Any, float], truth: set) -> dict[str, float]:
    """Threshold-free comparison: max F1 over all score cut-offs (separates ranking quality from calibration)."""
    best = {"f1": 0.0, "threshold": 1.0, "precision": 0.0, "recall": 0.0}
    for t in sorted(set(scores.values()), reverse=True):
        pred = {k for k, v in scores.items() if v >= t}
        m = prf(pred, truth)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "threshold": t, "precision": m["precision"], "recall": m["recall"]}
    return best


class PeakMemory:
    """Samples process RSS in a background thread (captures native DuckDB/Arrow memory too)."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return round(self.peak / 1e6, 1)

    @property
    def delta_mb(self) -> float:
        return round((self.peak - self.base) / 1e6, 1)


@contextmanager
def timed(store: dict, key: str):
    t = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - t, 3)


def save_json(name: str, obj: Any) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def markdown_table(rows: list[dict], columns: list[str], headers: list[str] | None = None, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(format(v, fmt[c]) if c in fmt and isinstance(v, (int, float)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _style(ax, title: str, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def grouped_bars(path: Path, groups: list[str], series: dict[str, list[float]], title: str, ylabel: str, ylim: tuple[float, float] | None = None, value_fmt: str = "{:.2f}") -> None:
    """Grouped bar chart with direct value labels (labels are required: some palette slots sit below 3:1 contrast)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(groups) * max(1, n / 2)), 4.2), dpi=150)
    for i, (name, values) in enumerate(series.items()):
        color = SERIES[i]
        for j, v in enumerate(values):
            x = j - 0.4 + width * (i + 0.5)
            bw = width - 0.03  # 2px-equivalent surface gap between adjacent bars
            ax.add_patch(FancyBboxPatch((x - bw / 2, 0), bw, v, boxstyle="round,pad=0,rounding_size=0.02", mutation_aspect=0.05, linewidth=0, facecolor=color, label=name if j == 0 else None))
            ax.text(x, v, value_fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    top = max((max(v) for v in series.values()), default=1)
    ax.set_ylim(*(ylim or (0, top * 1.15)))
    ax.set_xticks(range(len(groups)), groups)
    _style(ax, title, ylabel)
    if n >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def line_chart(path: Path, x: list[float], series: dict[str, list[float]], title: str, xlabel: str, ylabel: str, value_fmt: str = "{:.1f}",
               xticklabels: list[str] | None = None, ylim: tuple[float, float] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    direct = len(series) <= 4  # direct end labels only for few series; the legend carries identity otherwise
    fig, ax = plt.subplots(figsize=(7.5, 4.2) if direct else (8.5, 4.8), dpi=150)
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x, ys, color=SERIES[i], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        if direct:
            ax.annotate(f"{name}  {value_fmt.format(ys[-1])}", (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    _style(ax, title, ylabel)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_xticks(x, xticklabels if xticklabels is not None else [f"{int(v / 1000)}K" if v < 1e6 else f"{v / 1e6:g}M" for v in x])
    if direct:
        ax.set_xlim(min(x) - 0.2 * (max(x) - min(x)) if xticklabels is not None else min(x) * 0.8, max(x) + 0.35 * (max(x) - min(x)) if xticklabels is not None else max(x) * 1.35)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    if len(series) >= 2:
        if direct:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
        else:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
