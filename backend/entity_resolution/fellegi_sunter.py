"""Fellegi–Sunter probabilistic record linkage with EM parameter estimation.

For a record pair with comparison vector γ = (γ_1..γ_K) (one agreement level
per field), the match weight is the log-likelihood ratio

    W(γ) = log2( λ / (1−λ) ) + Σ_k log2( m_k(γ_k) / u_k(γ_k) )

    m_k(ℓ) = P(γ_k = ℓ | match)       u_k(ℓ) = P(γ_k = ℓ | non-match)

and P(match | γ) = 1 / (1 + 2^−W). Following common practice (e.g. Splink):

* **u** is estimated from uniformly random record pairs, which are almost all
  non-matches — a stable, unsupervised estimate;
* **m** is estimated by Expectation–Maximisation, but *not* naively over all
  blocked pairs: blocked candidate sets are dominated by look-alike
  non-matches (same surname), and plain EM then converges to a degenerate
  latent class ("shares a surname"). Instead, for every deterministic rule r
  (exact e-mail, exact phone, name key…) EM is run on the pairs satisfying r
  using only the *other* fields, and a field's m is averaged over the runs in
  which it was not the rule. This breaks the circularity between the blocking
  field and its own parameters (the training strategy popularised by Splink);
* the prior **λ** is finally re-estimated over all candidates with m, u fixed.

Laplace smoothing avoids zero probabilities on small samples.

A ``null`` level (missing value on either side) is neutral: m = u = 1.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.entity_resolution.comparators import LEVELS, NULL

Comparator = Callable[[Any, Any], tuple[str, float]]

# prior m-probabilities used to initialise EM
_M_INIT = {"exact": 0.80, "high": 0.10, "medium": 0.05, "different": 0.05}


@dataclass
class FSModel:
    m: dict[str, dict[str, float]] = field(default_factory=dict)
    u: dict[str, dict[str, float]] = field(default_factory=dict)
    prior: float = 0.01
    iterations: int = 0
    converged: bool = False

    def llr(self, levels: dict[str, str], u_exact: dict[str, float] | None = None) -> float:
        """Sum over fields of log2 m/u (bits). ``u_exact`` overrides u(exact) per field (term-frequency adjustment)."""
        w = 0.0
        for f, lvl in levels.items():
            if lvl == NULL:
                continue
            u = u_exact[f] if (u_exact and lvl == "exact" and f in u_exact) else self.u[f][lvl]
            w += math.log2(self.m[f][lvl] / u)
        return w

    def weight(self, levels: dict[str, str], u_exact: dict[str, float] | None = None) -> float:
        lam = min(max(self.prior, 1e-12), 1 - 1e-6)
        return math.log2(lam / (1 - lam)) + self.llr(levels, u_exact)

    def probability(self, levels: dict[str, str], u_exact: dict[str, float] | None = None) -> tuple[float, float]:
        w = self.weight(levels, u_exact)
        w = max(-60.0, min(60.0, w))
        return 1.0 / (1.0 + 2.0 ** (-w)), w

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_runs": getattr(self, "training_runs", []),
            "prior": round(self.prior, 6),
            "iterations": self.iterations,
            "converged": self.converged,
            "fields": {
                f: {lvl: {"m": round(self.m[f][lvl], 4), "u": round(self.u[f][lvl], 4), "bayes_factor": round(self.m[f][lvl] / self.u[f][lvl], 3)} for lvl in LEVELS}
                for f in self.m
            },
        }


def prior_over_all_pairs(llrs: list[float], total_pairs: int, start: float, iterations: int = 50, tol: float = 1e-9) -> float:
    """Prior lambda over *all* comparable pairs, consistent with u (estimated from random pairs of the whole population).

    Blocking keeps only candidate pairs; pairs outside the blocks are treated as non-matches (p ~ 0).
    Fixed point of  lambda = (1/T) * sum_i sigmoid2(logit2(lambda) + LLR_i)  over candidates, T = comparable pairs.
    Pairing a blocked-population prior with an all-pairs u inflates posterior odds by about T / |candidates|.
    """
    if total_pairs <= 0 or not llrs:
        return start
    lam = min(max(start, 1.0 / total_pairs), 0.5)
    for _ in range(iterations):
        lo = math.log2(lam / (1 - lam))
        expected = sum(1.0 / (1.0 + 2.0 ** (-max(-60.0, min(60.0, lo + x)))) for x in llrs)
        new = min(max(expected / total_pairs, 1.0 / total_pairs), 0.5)
        if abs(new - lam) < tol:
            return new
        lam = new
    return lam


def term_frequencies(records: list[dict[str, Any]], field_name: str, key: Callable[[Any], Any]) -> dict[Any, float]:
    """Relative frequency of each normalised value of a field among the records where it is present."""
    counts: dict[Any, int] = defaultdict(int)
    n = 0
    for r in records:
        v = r.get(field_name)
        k = key(v) if v is not None else None
        if k is None or k == "":
            continue
        counts[k] += 1
        n += 1
    return {k: c / n for k, c in counts.items()} if n else {}


def compare_pair(ra: dict[str, Any], rb: dict[str, Any], comparators: dict[str, Comparator]) -> tuple[dict[str, str], dict[str, float]]:
    levels, scores = {}, {}
    for f, cmp in comparators.items():
        lvl, sim = cmp(ra.get(f), rb.get(f))
        levels[f] = lvl
        scores[f] = round(sim, 4)
    return levels, scores


def estimate_u(records: list[dict[str, Any]], comparators: dict[str, Comparator], samples: int = 4000, seed: int = 13) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    counts: dict[str, dict[str, float]] = {f: {lvl: 1.0 for lvl in LEVELS} for f in comparators}  # Laplace
    n = len(records)
    if n >= 2:
        for _ in range(samples):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            levels, _ = compare_pair(records[i], records[j], comparators)
            for f, lvl in levels.items():
                if lvl != NULL:
                    counts[f][lvl] += 1
    return {f: {lvl: c / sum(cs.values()) for lvl, c in cs.items()} for f, cs in counts.items()}


def fit_em(
    pair_levels: list[dict[str, str]],
    u: dict[str, dict[str, float]],
    fields: list[str],
    prior: float,
    iterations: int = 25,
    tol: float = 1e-4,
) -> FSModel:
    model = FSModel(m={f: dict(_M_INIT) for f in fields}, u=u, prior=prior)
    if not pair_levels:
        return model
    # group identical comparison vectors: EM cost is O(#patterns) not O(#pairs)
    patterns: dict[tuple, int] = defaultdict(int)
    for lv in pair_levels:
        patterns[tuple(lv[f] for f in fields)] += 1
    items = list(patterns.items())
    for it in range(1, iterations + 1):
        m_counts = {f: {lvl: 0.5 for lvl in LEVELS} for f in fields}
        total_p = 0.0
        total_n = 0
        for pattern, count in items:
            levels = dict(zip(fields, pattern))
            p, _ = model.probability(levels)
            total_p += p * count
            total_n += count
            for f, lvl in levels.items():
                if lvl != NULL:
                    m_counts[f][lvl] += p * count
        new_m = {f: {lvl: c / sum(cs.values()) for lvl, c in cs.items()} for f, cs in m_counts.items()}
        new_prior = min(max(total_p / total_n, 1e-4), 0.99)
        delta = max(abs(new_m[f][l] - model.m[f][l]) for f in fields for l in LEVELS)
        delta = max(delta, abs(new_prior - model.prior))
        model.m, model.prior, model.iterations = new_m, new_prior, it
        if delta < tol:
            model.converged = True
            break
    # an "agreement" level must never be evidence *against* a match
    for f in fields:
        for lvl in ("exact", "high"):
            if model.m[f][lvl] < model.u[f][lvl]:
                model.m[f][lvl] = model.u[f][lvl]
    return model


def fit_label_calibration(weights: list[float], labels: list[bool], all_weights: list[float] | None = None, label_weight: float = 20.0) -> tuple[float, float] | None:
    """Semi-supervised recalibration of the Fellegi–Sunter match weight (weighted Platt scaling).

    Uncalibrated, P(match) = σ(ln2 · W) with W in bits. Actively selected labels are a
    biased sample (they sit near the decision boundary), so fitting a calibration to them
    alone distorts every probability. Instead we fit P(match) = σ(a·W + b) on **all**
    compared pairs, using the model's own probabilities as soft targets (weight 1 each),
    plus the labelled pairs as hard targets with weight ``label_weight``:

        min_{a,b}  Σ_all  CE(σ(ln2·W_i), σ(a·W_i + b))  +  λ Σ_labelled CE(y_j, σ(a·W_j + b))

    Needs at least one labelled match and one labelled non-match.
    """
    if not labels or all(labels) or not any(labels):
        return None
    import numpy as np
    from scipy.optimize import minimize

    a0 = math.log(2)
    wl = np.array(weights, dtype=float)
    yl = np.array(labels, dtype=float)
    wa = np.array(all_weights if all_weights is not None else weights, dtype=float)
    ya = 1 / (1 + np.exp(-np.clip(a0 * wa, -40, 40)))
    w_all = np.concatenate([wa, wl])
    y_all = np.concatenate([ya, yl])
    sw = np.concatenate([np.ones(len(wa)), np.full(len(wl), label_weight)])

    def loss(theta):
        a, b = theta
        p = 1 / (1 + np.exp(-np.clip(a * w_all + b, -40, 40)))
        eps = 1e-9
        return -np.sum(sw * (y_all * np.log(p + eps) + (1 - y_all) * np.log(1 - p + eps))) / sw.sum()

    res = minimize(loss, x0=[a0, 0.0], method="L-BFGS-B", bounds=[(0.01, 5.0), (-30.0, 30.0)])
    return float(res.x[0]), float(res.x[1])


def fit_rule_blocked_em(
    pair_levels: list[dict[str, str]],
    u: dict[str, dict[str, float]],
    fields: list[str],
    rule_fields: list[str],
    prior: float,
    iterations: int = 25,
    min_pairs: int = 20,
    labelled_matches: list[dict[str, str]] | None = None,
    label_weight: float = 5.0,
) -> FSModel:
    sums: dict[str, dict[str, float]] = {f: {lvl: 0.0 for lvl in LEVELS} for f in fields}
    weights: dict[str, float] = {f: 0.0 for f in fields}
    runs = []
    for rule in rule_fields:
        subset = [lv for lv in pair_levels if lv.get(rule) in ("exact", "high")]
        others = [f for f in fields if f != rule]
        if len(subset) < min_pairs or not others:
            continue
        sub_prior = min(0.9, max(prior, 0.5))
        model = fit_em([{f: lv[f] for f in others} for lv in subset], u, others, prior=sub_prior, iterations=iterations)
        runs.append({"rule": rule, "pairs": len(subset), "prior": round(model.prior, 4), "iterations": model.iterations})
        for f in others:
            for lvl in LEVELS:
                sums[f][lvl] += model.m[f][lvl] * len(subset)
            weights[f] += len(subset)
    # user-labelled true matches are direct evidence for m: each adds `label_weight` pseudo-pairs
    labelled_matches = labelled_matches or []
    if labelled_matches:
        runs.append({"rule": "user_labels", "pairs": len(labelled_matches), "weight": label_weight})
    final = FSModel(m={}, u=u, prior=prior)
    for f in fields:
        counts = {lvl: sum(1 for lv in labelled_matches if lv.get(f) == lvl) for lvl in LEVELS}
        n_lab = sum(counts.values())
        if weights[f] > 0 or n_lab:
            base = {lvl: (sums[f][lvl] / weights[f]) if weights[f] > 0 else _M_INIT[lvl] for lvl in LEVELS}
            w0 = weights[f] if weights[f] > 0 else 1.0
            total = w0 + label_weight * n_lab
            final.m[f] = {lvl: (base[lvl] * w0 + label_weight * counts[lvl] + 1e-3) / (total + 1e-3 * len(LEVELS)) for lvl in LEVELS}
        else:
            final.m[f] = dict(_M_INIT)
    for f in fields:
        for lvl in ("exact", "high"):
            if final.m[f][lvl] < final.u[f][lvl]:
                final.m[f][lvl] = final.u[f][lvl]
    # re-estimate the prior with m and u fixed
    patterns: dict[tuple, int] = defaultdict(int)
    for lv in pair_levels:
        patterns[tuple(lv[f] for f in fields)] += 1
    for it in range(1, iterations + 1):
        tot = n = 0.0
        for pattern, count in patterns.items():
            p, _ = final.probability(dict(zip(fields, pattern)))
            tot += p * count
            n += count
        new_prior = min(max(tot / max(n, 1), 1e-4), 0.99)
        final.iterations = it
        if abs(new_prior - final.prior) < 1e-5:
            final.prior = new_prior
            final.converged = True
            break
        final.prior = new_prior
    final.training_runs = runs  # type: ignore[attr-defined]
    return final
