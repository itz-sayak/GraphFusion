"""Client-side rate limiting for hosted LLMs (built around Groq's free tier).

Groq enforces four limits **per model** (console.groq.com/settings/limits):

    RPM  requests / minute      e.g. 30
    RPD  requests / day         e.g. 1,000   → header x-ratelimit-{limit,remaining,reset}-requests
    TPM  tokens / minute        e.g. 8,000   → header x-ratelimit-{limit,remaining,reset}-tokens
    TPD  tokens / day           e.g. 200,000 → NOT in any header; only visible in the 429 body

Measured on this project (Sep 2026): TPM is charged with actual usage
(prompt + completion), not the ``max_tokens`` reservation; one agent round with
all tool schemas costs ~3.5K prompt tokens. Waiting for 429s therefore wastes
the day: the limiter must *predict* and act before sending.

Design
------
* Sliding windows of (timestamp, tokens) per model: 60 s for RPM/TPM and 24 h
  for RPD/TPD (the day window is bucketed per minute to stay small).
* Every limit is applied with a safety margin (default 90%).
* Before a call: estimate tokens (chars / 3.2 of the request + expected output).
  - A **day** budget that would be exceeded → ``RateLimitExhausted`` immediately
    (never sleep for hours; the caller fails over to another model).
  - A **minute** budget that would be exceeded → sleep until enough of the window
    expires, if that is at most ``max_wait_s``; otherwise ``RateLimitExhausted``.
* After a call: record the provider-reported usage, and adopt the server's view of
  remaining TPM / RPD from the headers when it is stricter than ours.
* A 429 is parsed ("tokens per day (TPD)", "Please try again in 10m48s") and the
  model is blocked until then.
* State is persisted to ``<workspace>/_llm_usage.json``, so restarts, the API server
  and the benchmark scripts share one budget per model.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.errors import LLMProviderError
from backend.logging_conf import get_logger

log = get_logger(__name__)

MINUTE = 60.0
DAY = 86400.0


class RateLimitExhausted(LLMProviderError):
    """The request cannot be sent within the wait budget; ``retry_at`` is a wall-clock time."""

    def __init__(self, message: str, retry_at: float, scope: str):
        super().__init__(message)
        self.retry_at = retry_at
        self.scope = scope  # "minute" | "day"


@dataclass
class Limits:
    rpm: int = 30
    rpd: int = 1000
    tpm: int = 8000
    tpd: int = 200000

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Limits":
        d = d or {}
        return cls(**{k: int(d[k]) for k in ("rpm", "rpd", "tpm", "tpd") if k in d})


@dataclass
class _ModelState:
    minute: list[tuple[float, int]] = field(default_factory=list)  # (ts, tokens) per request, last 60 s
    day: dict[int, list[int]] = field(default_factory=dict)  # minute bucket → [requests, tokens], last 24 h
    blocked_until: float = 0.0
    blocked_reason: str = ""
    server_tpm_remaining: tuple[float, int] | None = None  # (valid until, remaining tokens)
    server_rpd_remaining: tuple[float, int] | None = None


def estimate_tokens(payload: Any, expected_output: int = 200) -> int:
    """Conservative token estimate for a JSON request body (≈ 3.2 chars per token for JSON-heavy prompts)."""
    text = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return int(len(text) / 3.2) + expected_output


def parse_duration(text: str) -> float:
    """'42.195s', '1m30s', '15m50.4s', '120ms', '2h3m' → seconds."""
    total = 0.0
    for value, unit in re.findall(r"([0-9.]+)(ms|h|m|s)", text or ""):
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


class RateLimiter:
    """Thread-safe, file-persisted limiter shared by all providers in the process."""

    def __init__(self, path: Path | None, limits: dict[str, dict[str, Any]] | None = None, safety: float = 0.9, max_wait_s: float = 20.0):
        self.path = path
        self.limits_cfg = limits or {}
        self.safety = safety
        self.max_wait_s = max_wait_s
        self._lock = threading.RLock()
        self._state: dict[str, _ModelState] = {}
        self._loaded_mtime = 0.0

    # ------------------------------------------------------------------ public API
    def limits(self, model: str) -> Limits:
        base = Limits.from_dict(self.limits_cfg.get("default"))
        over = self.limits_cfg.get(model) or {}
        return Limits(**{**base.__dict__, **{k: int(v) for k, v in over.items() if k in base.__dict__}})

    def acquire(self, model: str, est_tokens: int, max_wait_s: float | None = None) -> tuple[float, int]:
        """Block until the request fits the minute windows, or raise ``RateLimitExhausted``."""
        max_wait = self.max_wait_s if max_wait_s is None else max_wait_s
        deadline = time.monotonic() + max_wait
        while True:
            with self._lock:
                self._load()
                st = self._st(model)
                now = time.time()
                self._prune(st, now)
                lim = self.limits(model)
                if st.blocked_until > now:
                    raise RateLimitExhausted(f"{model} is rate-limited ({st.blocked_reason}) for another {_fmt(st.blocked_until - now)}", st.blocked_until, "day" if "day" in st.blocked_reason else "minute")
                day_req = sum(b[0] for b in st.day.values())
                day_tok = sum(b[1] for b in st.day.values())
                if day_tok + est_tokens > lim.tpd * self.safety:
                    retry = self._day_release_time(st, day_tok + est_tokens - lim.tpd * self.safety, now)
                    raise RateLimitExhausted(f"{model}: daily token budget nearly used ({day_tok:,} of {lim.tpd:,} tokens in the last 24 h)", retry, "day")
                rpd_left = lim.rpd * self.safety - day_req
                if st.server_rpd_remaining and st.server_rpd_remaining[0] > now:
                    rpd_left = min(rpd_left, st.server_rpd_remaining[1] - lim.rpd * (1 - self.safety))
                if rpd_left < 1:
                    raise RateLimitExhausted(f"{model}: daily request budget used ({day_req} requests in the last 24 h)", now + 3600, "day")
                wait = self._minute_wait(st, lim, est_tokens, now)
                if wait <= 0:
                    st.minute.append((now, est_tokens))  # provisional; replaced by actual usage in record()
                    self._bucket(st, now)[0] += 1
                    self._bucket(st, now)[1] += est_tokens
                    self._save()
                    return (now, est_tokens)
            if time.monotonic() + wait > deadline:
                raise RateLimitExhausted(f"{model}: per-minute limit reached; next slot in {_fmt(wait)}", time.time() + wait, "minute")
            log.info("Pacing for LLM rate limit", model=model, wait_s=round(wait, 1), est_tokens=est_tokens)
            time.sleep(min(wait, max(0.1, deadline - time.monotonic())))

    def record(self, model: str, ticket: tuple[float, int], actual_tokens: int | None, headers: dict[str, str] | None = None) -> None:
        """Replace the provisional estimate (``ticket`` from ``acquire``) with provider-reported usage; sync with headers."""
        ts, est_tokens = ticket
        with self._lock:
            self._load()
            st = self._st(model)
            now = time.time()
            if actual_tokens is not None:
                for i, (t0, tok) in enumerate(st.minute):
                    if t0 == ts and tok == est_tokens:
                        st.minute[i] = (t0, actual_tokens)
                        break
                bucket = st.day.get(int(ts // 60))
                if bucket is not None:
                    bucket[1] = max(0, bucket[1] + actual_tokens - est_tokens)
            h = {k.lower(): v for k, v in (headers or {}).items()}
            try:
                if "x-ratelimit-remaining-tokens" in h:
                    st.server_tpm_remaining = (now + max(1.0, parse_duration(h.get("x-ratelimit-reset-tokens", "1s"))), int(float(h["x-ratelimit-remaining-tokens"])))
                if "x-ratelimit-remaining-requests" in h:
                    st.server_rpd_remaining = (now + 60.0, int(float(h["x-ratelimit-remaining-requests"])))
            except ValueError:
                pass
            self._save()

    def release(self, model: str, ticket: tuple[float, int]) -> None:
        """A request that never reached the provider (connection failed) does not count."""
        self.record(model, ticket, 0)
        with self._lock:
            bucket = self._st(model).day.get(int(ticket[0] // 60))
            if bucket is not None:
                bucket[0] = max(0, bucket[0] - 1)
            self._save()

    def block_from_429(self, model: str, body: str, headers: dict[str, str] | None = None) -> float:
        """Parse a 429 and block the model until the provider says it may be retried. Returns seconds."""
        text = body or ""
        m = re.search(r"try again in ([0-9hms.]+)", text)
        secs = parse_duration(m.group(1)) if m else 0.0
        ra = (headers or {}).get("retry-after")
        if not secs and ra:
            try:
                secs = float(ra)
            except ValueError:
                pass
        per_day = bool(re.search(r"per day|\(TPD\)|\(RPD\)", text))
        reason = ("tokens per day" if "TPD" in text else "requests per day" if "RPD" in text else
                  "tokens per minute" if "TPM" in text else "requests per minute" if "RPM" in text else "rate limit")
        secs = secs or (600.0 if per_day else 20.0)
        with self._lock:
            self._load()
            st = self._st(model)
            st.blocked_until = max(st.blocked_until, time.time() + secs + 1.0)
            st.blocked_reason = reason
            self._save()
        log.warning("LLM rate limit hit", model=model, reason=reason, blocked_s=round(secs, 1))
        return secs

    def usage(self, model: str) -> dict[str, Any]:
        with self._lock:
            self._load()
            st = self._st(model)
            now = time.time()
            self._prune(st, now)
            lim = self.limits(model)
            return {
                "model": model, "limits": lim.__dict__,
                "minute": {"requests": len(st.minute), "tokens": sum(t for _, t in st.minute)},
                "day": {"requests": sum(b[0] for b in st.day.values()), "tokens": sum(b[1] for b in st.day.values())},
                "blocked_until": st.blocked_until if st.blocked_until > now else None, "blocked_reason": st.blocked_reason if st.blocked_until > now else None,
            }

    def headroom(self, model: str, est_tokens: int) -> bool:
        """True if a request of this size could be sent right now without waiting."""
        with self._lock:
            self._load()
            st = self._st(model)
            now = time.time()
            self._prune(st, now)
            lim = self.limits(model)
            day_tok = sum(b[1] for b in st.day.values())
            return st.blocked_until <= now and day_tok + est_tokens <= lim.tpd * self.safety and self._minute_wait(st, lim, est_tokens, now) <= 0

    # ------------------------------------------------------------------ internals
    def _minute_wait(self, st: _ModelState, lim: Limits, est: int, now: float) -> float:
        tpm_cap = lim.tpm * self.safety
        rpm_cap = max(1, int(lim.rpm * self.safety))
        used = sum(t for _, t in st.minute)
        waits = [0.0]
        if len(st.minute) + 1 > rpm_cap:
            waits.append(st.minute[len(st.minute) - rpm_cap][0] + MINUTE - now)
        if est > tpm_cap:
            # larger than the whole window: send when the window is empty
            waits.append((st.minute[-1][0] + MINUTE - now) if st.minute else 0.0)
        elif used + est > tpm_cap:
            need = used + est - tpm_cap
            freed = 0
            for ts, t in st.minute:
                freed += t
                if freed >= need:
                    waits.append(ts + MINUTE - now)
                    break
        if st.server_tpm_remaining and st.server_tpm_remaining[0] > now and st.server_tpm_remaining[1] < est:
            waits.append(st.server_tpm_remaining[0] - now)
        return max(waits)

    @staticmethod
    def _day_release_time(st: _ModelState, need: float, now: float) -> float:
        freed = 0
        for minute in sorted(st.day):
            freed += st.day[minute][1]
            if freed >= need:
                return minute * 60 + DAY + 60
        return now + DAY

    @staticmethod
    def _bucket(st: _ModelState, now: float) -> list[int]:
        return st.day.setdefault(int(now // 60), [0, 0])

    @staticmethod
    def _prune(st: _ModelState, now: float) -> None:
        st.minute = [(ts, t) for ts, t in st.minute if now - ts < MINUTE]
        cutoff = int((now - DAY) // 60)
        for k in [k for k in st.day if k <= cutoff]:
            del st.day[k]

    def _st(self, model: str) -> _ModelState:
        return self._state.setdefault(model, _ModelState())

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            mtime = self.path.stat().st_mtime
            if mtime <= self._loaded_mtime:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._state = {m: _ModelState(minute=[tuple(x) for x in d.get("minute", [])], day={int(k): list(v) for k, v in d.get("day", {}).items()},
                                          blocked_until=d.get("blocked_until", 0.0), blocked_reason=d.get("blocked_reason", ""))
                           for m, d in raw.items()}
            self._loaded_mtime = mtime
        except (OSError, ValueError, TypeError):
            log.warning("Could not read LLM usage file; starting a fresh window", path=str(self.path))

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {m: {"minute": st.minute, "day": st.day, "blocked_until": st.blocked_until, "blocked_reason": st.blocked_reason} for m, st in self._state.items()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.path)
            self._loaded_mtime = self.path.stat().st_mtime
        except OSError:
            log.warning("Could not persist LLM usage", path=str(self.path))


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


_shared: RateLimiter | None = None


def shared_limiter() -> RateLimiter:
    global _shared
    if _shared is None:
        from backend.config import get_settings, load_config

        cfg = (load_config().get("llm") or {}).get("rate_limits") or {}
        _shared = RateLimiter(Path(get_settings().dfg_workspace) / "_llm_usage.json", cfg.get("models") or {},
                              safety=float(cfg.get("safety_margin", 0.9)), max_wait_s=float(cfg.get("max_wait_seconds", 20)))
    return _shared
