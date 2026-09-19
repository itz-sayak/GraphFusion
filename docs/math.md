# Mathematical formulation

## Integration graph

```text
G = (V, E),   V = D ∪ C ∪ N ∪ O        datasets, columns, entities, outputs
E ⊆ V × V × R                          R = {contains, similar_column, foreign_key_candidate, value_overlap,
                                             semantic_match, dataset_relationship, same_entity, derived_from}
c : E → [0, 1]                         confidence
cost(e) = −log c(e)    (default)   or   1 − c(e)   (config: graph.path_cost)
```

## Shortest paths

For a path `P = (e₁, …, e_k)`, treating edge confidences as independent evidence, the path reliability is

```text
ρ(P) = Π_i c(e_i)
```

Since `−log` is strictly decreasing and `−log c ≥ 0` for `c ∈ (0, 1]`:

```text
P* = argmin_P Σ_i −log c(e_i) = argmax_P ρ(P)
```

Dijkstra is exact for non-negative costs, so it returns the maximum-reliability route in `O((|V| + |E|) log |V|)`.

With linear costs `Σ (1 − c_i)`, ranking does not follow reliability. Three hops of 0.8 cost 0.60 (ρ = 0.512), while a direct 0.45 edge costs 0.55 (ρ = 0.45). Linear costs choose the direct edge; `−log` costs (0.669 vs 0.799) choose the more reliable path. `tests/test_graph.py::test_neglog_and_linear_costs_disagree_where_documented` asserts this.

## Chance overlap of integer keys

For a key column $`t`$ dense on $`[l, h]`$ ($`\text{distinct}(t) \ge 0.9\,(h - l + 1)`$) and a reference $`r`$ spread over $`[l_r, h_r]`$, the containment expected from the ranges alone is

```math
\mathbb{E}[\text{cont}(r, t)] = \frac{\max\big(0,\ \min(h, h_r) - \max(l, l_r) + 1\big)}{h_r - l_r + 1}
```

An observed containment $`\le \mathbb{E} + 0.05`$ carries no evidence, and the link then needs name support.

## Root choice

With $`T`$ the spanning tree and $`\mathrm{keeps}(u \to v)`$ true for same-entity joins and for lookups whose fact side is $`u`$,

```math
\mathrm{root} = \arg\max_{d}\ \Big( \big|\{v : \text{every edge on the } T\text{-path } d \leadsto v \text{ keeps the grain}\}\big|,\ [d \notin \text{dims}],\ \text{rows}(d) \Big)
```

(lexicographic). Every other dataset is aggregated on the way, so the root minimises the number of summarised tables.

## Merge structure

Among the relationships above the mode threshold, the join structure is the maximum spanning forest

```text
T* = argmax_{T spanning forest} Σ_{e∈T} c(e)
```

A tree has exactly `n − 1` edges per component. Adding any excluded edge closes a cycle, i.e. two routes between the same datasets, which would join the same rows twice. Among acyclic structures, maximising total confidence prefers reliable joins, and Kruskal solves this exactly in `O(|E| log |E|)`.

## Schema-matching score

```text
S(ci, cj) = ( w_N·N + w_T·T + w_E·E + w_V·V + w_D·D + w_P·P + w_C·C ) / Σ w
```

| Term | Definition |
|---|---|
| N | `max( ME_sym(t_i, t_j), 0.9·JW(joined) if > .85, 0.95·TSR if > .8 )`, ME = weighted Monge–Elkan over expanded tokens, inner JW (values < .8 halved), generic tokens weight 0.5 |
| T | type lattice (1, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.1) |
| E | `0.4·Jaccard(concepts) + 0.3·semtype_compat + 0.3·cos(embedding)` |
| V | `f · (0.4·J + 0.6·max(κ_ij, κ_ji))`, normaliser `ν* = argmax_ν (0.4·J_ν + 0.6·κ_ν) − cost(ν)` |
| D | `1 − mean_q |Q_i(q) − Q_j(q)| / range` over q ∈ {5,25,50,75,95}, or length similarity |
| P | cosine of shape histograms |
| C | `√(min(d_i, d_j) / max(d_i, d_j))` |

with containment `κ_ij = |A_i ∩ A_j| / |A_i|`. For integer domains `V` uses the chance-adjusted containment

```text
e_ij = |{v ∈ A_j : min A_i ≤ v ≤ max A_i}| / (max A_i − min A_i + 1)
κ*_ij = max(0, (κ_ij − e_ij) / (1 − e_ij))   (0 if e_ij ≈ 1)
V = f · (0.5·V_raw + 0.5·max(κ*_ij, κ*_ji))
```

and the small-domain factor `f = min(1, log₂(1 + min|A|) / log₂ 33)` (textual domains: `0.8 + 0.2f`). If both semantic types are confidently known and incompatible, `S ← 0.6·S`.

## Similarity flooding (multi-dataset adaptation)

For pair `(a, b)`, `a ∈ A`, `b ∈ B`, base score `s⁰`:

```text
T^k(a,b) = max_{c ∈ X, X ∉ {A,B}} s^k(a,c) · s^k(c,b)
Σ^k(a,b) = mean of top-3 { max_{b' ≠ b} s^k(a',b') : a' ∈ A \ {a} }
X^k(a,b) = s^k(a,b) / max( max_{b'} s^k(a,b'), max_{a'} s^k(a',b) )
s^{k+1}  = clip₀₁( s⁰ + α·max(0, T^k − s⁰) + β·max(0, Σ^k − μ) ) · (X^k)^γ
```

with α = 0.5, β = 0.15, μ = 0.6, γ = 0.35. Iteration stops at `max |Δ| < 10⁻³` or 10 rounds.

## Bipartite alignment

For dataset pair (A, B), with `x_ab ∈ {0,1}` and each column used at most once:

```text
max Σ_{a,b} S(a,b)·x_ab    s.t.  Σ_b x_ab ≤ 1,  Σ_a x_ab ≤ 1,  S(a,b) ≥ τ
```

It is solved with the Hungarian algorithm. A pair is re-admitted when `uniq(b) ≥ 0.98 ∧ κ_ab ≥ 0.9` (foreign key).

## Fellegi–Sunter

For a comparison vector `γ` with levels `ℓ_k`:

```text
m_k(ℓ) = P(γ_k = ℓ | M),   u_k(ℓ) = P(γ_k = ℓ | U)
W(γ)   = log₂(λ / (1 − λ)) + Σ_k log₂( m_k(ℓ_k) / u_k(ℓ_k) )       (null levels contribute 0)
P(M | γ) = 1 / (1 + 2^{−W})
```

EM for a rule r uses pairs `R_r = {γ : γ_r ∈ {exact, high}}` and fields `F \ {r}`:

```text
E:  p_γ = P(M | γ_{F\r})
M:  m_k(ℓ) = (Σ_γ p_γ·1[γ_k = ℓ] + ½) / (Σ_γ p_γ + 2),     λ = mean p_γ
```

Final `m_k` averages the runs with `k ≠ r` (weighted by `|R_r|`); `m_k(ℓ) ≥ u_k(ℓ)` is enforced for agreement levels.

## Correlation clustering

Edge weights `w_ij = logit(p_ij) − logit(t)` for compared pairs, and `w₀ = −0.4` for uncompared pairs:

```text
max_{partition 𝒞} Σ_{C ∈ 𝒞} Σ_{i<j ∈ C} w_ij
```

The problem is NP-hard; greedy additive edge contraction merges the cluster pair with the largest positive `W(C₁,C₂) = Σ w_ij + n_uncompared·w₀` until none remains.

## Truth discovery (source_accuracy_vote)

```text
score(v) = Σ_{s claims v} log( a_s / (1 − a_s) )
a_s ← (agreements_s + 1) / (claims_s + 2)        iterate to convergence
```

## Learned matcher

```math
\hat p(x) = \mathrm{Iso}\big(f_{\mathrm{GBDT}}(x)\big), \qquad
\min_f \sum_{i \in \text{base}} \ell(y_i, f(x_i)) + \eta \sum_{j \in \text{decisions}} \ell(y_j, f(x_j)), \quad \eta = 10
```

## Active learning

Margin uncertainty around the auto-merge threshold $`t`$, with a disagreement bonus:

```math
u_{ij} = \exp\big(-\lvert \mathrm{logit}(p_{ij}) - \mathrm{logit}(t) \rvert\big) + 0.5\,\mathbb{1}\big[\text{same cluster} \ne (p_{ij} \ge t)\big]
```

Constraints in correlation clustering: $`w_{ij} = +10^6`$ (must-link), $`-10^6`$ (cannot-link). Semi-supervised recalibration on randomly sampled labels $`\mathcal{L}_{\text{rand}}`$:

```math
\min_{a,b} \sum_{i} \mathrm{CE}\big(\sigma(\ln 2\, W_i), \sigma(a W_i + b)\big) + \lambda \sum_{j \in \mathcal{L}_{\text{rand}}} \mathrm{CE}\big(y_j, \sigma(a W_j + b)\big), \quad \lambda = 20
```

## Temporal validity

For candidates of entity $`e`$, attribute $`a`$ sorted by timestamp, with runs $`R_1, \dots, R_m`$ of equal normalised value starting at $`t_1 < \dots < t_m`$:

```math
\mathrm{valid}(e, a, \tau) = v_k \iff t_k \le \tau < t_{k+1} \quad (t_{m+1} = \infty)
```

Transaction time: with merges $`m`$ recorded at $`\mathrm{rec}(m)`$ and $`H_m`$ the history each wrote,

```math
\mathrm{known}(e, a, \tau, t) = \mathrm{valid}_{H_{m^*}}(e, a, \tau), \qquad m^* = \arg\max_{m:\ \mathrm{rec}(m) \le t} \mathrm{rec}(m)
```

## Uncertainty propagation

With $`Z_i \sim \mathrm{Bernoulli}(p_i)`$ independent and $`S^* = \sum_{i \in G} Z_i x_i`$:

```math
\mathbb{E}[S^*] = \sum_{i \in G} p_i x_i, \qquad \mathrm{Var}[S^*] = \sum_{i \in G} p_i(1-p_i) x_i^2, \qquad \text{CI}_{95} = \mathbb{E}[S^*] \pm 1.96\sqrt{\mathrm{Var}[S^*]}
```

Schema-level confidence $`r_i = \prod_{j \in \text{lookups}(i)} c_j`$ is shared across rows (a single Bernoulli per relationship), so it is reported separately and not added to the variance. For an uncertain relationship $`R`$ with confidence $`r_R`$ on the lineage path of the aggregated columns, the scenario value is $`S_G^{\neg R} = \sum_{i \in G \setminus J_R} x_i`$ with probability $`1 - r_R`$, where $`J_R`$ are the rows joined through $`R`$. Calibration is measured by expected calibration error over $`B`$ equal-mass bins:

```math
\mathrm{ECE} = \sum_{b=1}^{B} \frac{\lvert B_b \rvert}{n} \Big\lvert \overline{p}_{B_b} - \overline{y}_{B_b} \Big\rvert
```

## Calibrated prior over all pairs

```math
\frac{P(M\mid\gamma)}{P(U\mid\gamma)} = \frac{\lambda}{1-\lambda}\prod_k \frac{m_k(\gamma_k)}{u_k(\gamma_k)}, \qquad
\lambda = \frac{1}{T}\sum_{i\in C}\sigma_2\big(\operatorname{logit}_2\lambda + \mathrm{LLR}_i\big)
```

$`u`$ comes from random pairs of the whole population, so $`\lambda`$ must refer to the same population ($`T`$ comparable pairs, non-candidates treated as non-matches). A prior estimated over the blocked candidates $`C`$ inflates the odds by about $`T/\lvert C\rvert`$.

Term-frequency variant (measured, off by default): $`u_k(\text{exact}\mid v) = \mathrm{freq}_k(v)`$.

Field dependence (checked before adding log-linear interaction terms):

```math
\mathrm{lift}(j,k) = \frac{P(\gamma_j = \text{exact}, \gamma_k = \text{exact} \mid M)}{P(\gamma_j = \text{exact}\mid M)\,P(\gamma_k = \text{exact}\mid M)}
```

Measured 0.996–1.000 for all field pairs, i.e. conditional independence holds and interaction terms would not change the posterior.

## Value crosswalks

For a proposed mapping $`M: V_A \to V_B \cup \{\bot\}`$ with $`D = \{a : M(a) \in V_B\}`$:

```math
\frac{\lvert\{a : M(a) \notin V_B \cup \{\bot\}\}\rvert}{\lvert\{a : M(a) \ne \bot\}\rvert} \le 0.2, \qquad \frac{\lvert D \rvert}{\lvert V_A \rvert} \ge 0.5, \qquad \frac{\lvert M(D) \rvert}{\lvert D \rvert} \ge 0.8
```

An accepted $`M`$ becomes the value normaliser $`\nu_M`$ for that column pair, and value overlap is computed on $`\nu_M(V_A)`$ vs $`V_B`$.

## Adaptive blocking

A token block $`B`$ yields $`\binom{\lvert B\rvert}{2}`$ pairs. If $`\lvert B \rvert > L`$ it is partitioned by a secondary key $`g`$ into $`B = \bigsqcup_v B_v`$, giving $`\sum_v \binom{\lvert B_v\rvert}{2} \le \binom{\lvert B\rvert}{2}`$ pairs; with $`c`$ roughly equal groups the block's cost drops by about $`c`$×. Pairs across groups are lost only if they share no other blocking key (e-mail, phone, id, sorted neighbourhood).

