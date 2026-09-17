"""Graph-based refinement of column similarities (similarity flooding).

Classic similarity flooding (Melnik, Garcia-Molina & Rahm, ICDE'02) propagates
pairwise similarity over a *pairwise connectivity graph* until a fixpoint:
a pair gains similarity when the pairs adjacent to it are similar. Flat tables
offer little intra-schema structure, so we adapt the propagation graph to the
multi-dataset integration setting. For a candidate pair (a in A, b in B) at
iteration k:

* transitive support   T^k(a,b) = max over c in a third dataset of s^k(a,c) * s^k(c,b)
  -- reliability of the best two-hop route (the same multiplicative path
  semantics used by Dijkstra on -log c);
* structural support   Sigma^k(a,b) = mean of the top-3 best sibling alignments
  between A minus a and B minus b -- tables that align well elsewhere make a
  candidate more plausible (the flooding intuition);
* exclusivity          X^k(a,b) = s^k(a,b) / max(best_a, best_b)
  -- competition normalisation: a pair dominated by a better alternative for
  either column is damped.

    s^{k+1} = clip01( s0 + alpha*max(0, T^k - s0) + beta*max(0, Sigma^k - mu) ) * (X^k)^gamma

Supports only *raise* the base score (a dimension table legitimately has a
single matching column, so weak structure must not penalise it); only
exclusivity lowers it. Iteration stops when max |delta| < epsilon or after K rounds.
"""
from __future__ import annotations

from collections import defaultdict

from backend.core.models import MatchEvidence


def flood(
    base: dict[tuple[str, str], float],
    dataset_of: dict[str, str],
    evidence: dict[tuple[str, str], MatchEvidence],
    params: dict,
) -> tuple[dict[tuple[str, str], float], dict]:
    alpha, beta = params["transitive_alpha"], params["structural_beta"]
    center, gamma = params["structural_center"], params["exclusivity_gamma"]
    eps, max_iter = params["epsilon"], params["max_iterations"]

    adj: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    for pair in base:
        a, b = pair
        adj[a][b] = pair
        adj[b][a] = pair

    columns_of: dict[str, list[str]] = defaultdict(list)
    for col in adj:
        columns_of[dataset_of[col]].append(col)

    s = dict(base)
    history: list[float] = []
    iterations = 0
    for iterations in range(1, max_iter + 1):
        # two best partners per (column, other dataset) -- used by Sigma and X
        best_by: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
        for (a, b), v in s.items():
            best_by[(a, dataset_of[b])].append((v, b))
            best_by[(b, dataset_of[a])].append((v, a))
        top2 = {k: sorted(v, reverse=True)[:2] for k, v in best_by.items()}

        new: dict[tuple[str, str], float] = {}
        delta = 0.0
        for pair, s0 in base.items():
            a, b = pair
            da, db = dataset_of[a], dataset_of[b]
            cur = s[pair]

            t = 0.0
            na, nb = adj[a], adj[b]
            small, large = (na, nb) if len(na) <= len(nb) else (nb, na)
            for c, p1 in small.items():
                if dataset_of[c] in (da, db) or c not in large:
                    continue
                t = max(t, s[p1] * s[large[c]])

            sib: list[float] = []
            for a2 in columns_of[da]:
                if a2 == a:
                    continue
                for v, partner in top2.get((a2, db), []):
                    if partner != b:
                        sib.append(v)
                        break
            sib.sort(reverse=True)
            sigma = sum(sib[:3]) / 3 if sib else 0.0

            best = max(top2.get((a, db), [(cur, b)])[0][0], top2.get((b, da), [(cur, a)])[0][0])
            x = cur / best if best > 0 else 1.0

            value = s0 + alpha * max(0.0, t - s0) + beta * max(0.0, sigma - center)
            value = min(1.0, max(0.0, value)) * (x ** gamma)
            new[pair] = value
            delta = max(delta, abs(value - cur))

            ev = evidence.get(pair)
            if ev is not None:
                ev.transitive_support = round(t, 4)
                ev.structural_support = round(sigma, 4)
                ev.exclusivity = round(x, 4)
        s = new
        history.append(round(delta, 6))
        if delta < eps:
            break

    for pair, ev in evidence.items():
        if pair in s:
            ev.graph_adjustment = round(s[pair] - base[pair], 4)
    return {k: round(v, 4) for k, v in s.items()}, {"iterations": iterations, "deltas": history, "converged": bool(history and history[-1] < eps)}
