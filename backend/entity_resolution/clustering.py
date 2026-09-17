"""From pairwise match probabilities to entity clusters.

**Why not connected components?** Taking the transitive closure of every
link above a threshold chains errors: A~B (0.95) and B~C (0.93) put A and C
in one entity even when A~C was explicitly compared and scored 0.01. On real
duplicate data this produces giant "snowball" clusters.

We instead solve a **correlation clustering** objective with Greedy Additive
Edge Contraction (GAEC; Keuper et al., ICCV'15), a strong heuristic for the
NP-hard problem:

    edge weight  w_ij = logit(p_ij)          (positive → evidence for same entity)
    cluster gain W(C1,C2) = Σ_{i∈C1, j∈C2, compared} w_ij  +  n_uncompared · w_default

Repeatedly contract the cluster pair with the largest positive gain, updating
aggregated gains, until no positive gain remains. An explicit strong
non-match between any members makes a contraction unattractive, which is
exactly the constraint transitive closure ignores. Uncompared pairs (different
blocks) contribute a small negative prior ``w_default``.

Connected components is kept as a baseline for the evaluation.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Hashable

import networkx as nx

Node = Hashable


def logit(p: float, cap: float = 12.0) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return max(-cap, min(cap, math.log(p / (1 - p))))


def connected_components_clusters(nodes: list[Node], edges: dict[tuple[Node, Node], float], threshold: float, constraints: dict | None = None) -> list[set[Node]]:
    constraints = constraints or {}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e, p in edges.items():
        label = constraints.get(e)
        if label is True or (label is None and p >= threshold):
            g.add_edge(*e)
    return [set(c) for c in nx.connected_components(g)]


def correlation_clusters(
    nodes: list[Node],
    edges: dict[tuple[Node, Node], float],
    threshold: float = 0.5,
    default_weight: float = -0.4,
    constraints: dict[tuple[Node, Node], bool] | None = None,
) -> list[set[Node]]:
    """GAEC correlation clustering. ``threshold`` shifts the logit so p = threshold is neutral.

    ``constraints`` are user labels: must-link pairs get weight +HARD and cannot-link pairs
    −HARD, large enough that no sum of ordinary evidence can override them.
    """
    HARD = 1e6
    constraints = constraints or {}
    shift = logit(threshold)
    idx = {n: i for i, n in enumerate(nodes)}
    members: dict[int, list[Node]] = {i: [n] for i, n in enumerate(nodes)}
    # adj[c1][c2] = (sum of weights over compared pairs, number of compared pairs)
    adj: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for (a, b), p in edges.items():
        ia, ib = idx[a], idx[b]
        if ia == ib:
            continue
        w = logit(p) - shift
        label = constraints.get((a, b), constraints.get((b, a)))
        if label is not None:
            w = HARD if label else -HARD
        for x, y in ((ia, ib), (ib, ia)):
            cell = adj[x].setdefault(y, [0.0, 0])
            cell[0] += w
            cell[1] += 1

    def gain(c1: int, c2: int) -> float:
        s, n = adj[c1].get(c2, (0.0, 0))
        uncompared = len(members[c1]) * len(members[c2]) - n
        return s + uncompared * default_weight

    heap: list[tuple[float, int, int]] = []
    for c1, nbrs in adj.items():
        for c2 in nbrs:
            if c1 < c2:
                g = gain(c1, c2)
                if g > 0:
                    heapq.heappush(heap, (-g, c1, c2))

    alive = set(members)
    while heap:
        neg_g, c1, c2 = heapq.heappop(heap)
        if c1 not in alive or c2 not in alive:
            continue
        g = gain(c1, c2)
        if abs(g + neg_g) > 1e-9:  # stale entry: re-queue with the current gain
            if g > 0:
                heapq.heappush(heap, (-g, c1, c2))
            continue
        if g <= 0:
            continue
        # contract c2 into c1
        members[c1].extend(members.pop(c2))
        alive.discard(c2)
        for c3, (s, n) in list(adj[c2].items()):
            if c3 == c1:
                continue
            cell = adj[c1].setdefault(c3, [0.0, 0])
            cell[0] += s
            cell[1] += n
            back = adj[c3].setdefault(c1, [0.0, 0])
            back[0] += s
            back[1] += n
            adj[c3].pop(c2, None)
        adj[c1].pop(c2, None)
        adj.pop(c2, None)
        for c3 in adj[c1]:
            if c3 in alive:
                g2 = gain(c1, c3)
                if g2 > 0:
                    a, b = (c1, c3) if c1 < c3 else (c3, c1)
                    heapq.heappush(heap, (-g2, a, b))
    return [set(members[c]) for c in alive]
