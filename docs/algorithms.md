# Algorithms

Every algorithm below is used for one specific job, with the reason it fits. Mathematical detail is in [math.md](math.md).

## 1. Profiling and semantic types (`backend/profiling/`)

* **Lossless CSV typing.** Columns are read as text; a column is promoted to BIGINT / DOUBLE / BOOLEAN / TIMESTAMP only when *every* value casts cleanly. Integers must match `^-?[0-9]+$`, because DuckDB's `TRY_CAST('11.5' AS BIGINT)` silently rounds. Leading zeros block numeric promotion. Sampling-based sniffers fail on mixed date formats; this approach does not.
* **Semantic types** (ID, NAME, EMAIL, PHONE, DATE, DATETIME, CITY, COUNTRY, ZIPCODE, CURRENCY, CURRENCY_CODE, LATITUDE, LONGITUDE, CATEGORY, FREE_TEXT, NUMERIC) combine regex/gazetteer rates on a reservoir sample, statistical signatures (range, uniqueness, length), physical type and name hints. Value evidence dominates, so `col_7` full of e-mails is EMAIL. Measure tokens (population, percent, median…) veto ID.
* **Coordinated bottom-k sketches.** DuckDB keeps the k distinct normalised values with the smallest `hash(value)`. Because all columns share the hash, samples are coordinated and overlap estimates stay unbiased (the KMV idea behind LSH Ensemble / JOSIE). They are exact when a column has ≤ k distinct values.
* **Day-first detection.** A column is day-first if any value only parses day-first (e.g. `23/04/2024`).
* **Malformed CSV repair** (`ingestion/csv_repair.py`). Triggered only when the sniffer reads a delimited file as one column. The delimiter is the candidate giving the header's field count on the most rows. Each column gets a value-class profile (int / num / date / text) from the well-formed rows. A short row is padded; a row with k extra fields is rebuilt by merging k+1 adjacent fields into one text column. Every merge position is scored 2·(fields fitting their column's class) + 1 + 0.5·(merged fields starting with a space), and ties are counted as ambiguous. Northwind: 209 rows rebuilt, 0 ambiguous.
* **Null markers.** `NULL` and `\N` (SQL exports) are read as missing values, so such columns still type as dates or integers.
* **SQLite.** Every table becomes its own dataset, named after the table.

## 2. Schema matching (`backend/matching/`)

1. **Candidates.** All cross-dataset pairs below 5 000 pairs. Above that, the union of MinHash **LSH Ensemble** containment queries, a name-token/thesaurus inverted index, and confident semantic-type buckets.
2. **Seven signals.** Name, type, semantic, values, distribution, pattern, cardinality (definitions in README §6).
3. **Data-driven value normalisation.** Every applicable normaliser (basic, alnum, identifier, digits, geo, reference domains, composite `semantic:<TYPE>`) is tried, and the one maximising `0.4·Jaccard + 0.6·containment` is kept. The winner becomes the join transformation.
4. **Chance-adjusted integer overlap.** For integer domains, the containment expected at random is the density of the target inside the source's range. Only the surplus `(c − e)/(1 − e)` counts. This removes false keys such as county FIPS ⊂ LocationID 1–265.
5. **Similarity flooding (adapted).** Flat tables have little intra-schema structure, so propagation runs over the *multi-dataset* pair graph. Two-hop support (`max_c s(a,c)·s(c,b)`) and structural support (best sibling alignments) can only raise a score; exclusivity `s/max(best_a, best_b)` damps dominated pairs. It iterates to a fixpoint (ε = 1e-3, ≤ 10 rounds).
6. **Bipartite alignment.** Hungarian assignment per dataset pair, followed by re-admission of pairs that reference a unique key with containment ≥ 0.9 (many-to-one foreign keys such as PULocationID/DOLocationID → LocationID).

## 3. Dataset relationships (`backend/graph/relationships.py`)

Aggregates accepted matches per dataset pair into one typed relationship:

| Kind | Condition | Confidence |
|---|---|---|
| LOOKUP | a keyish column (or verified FK) references a unique key (uniqueness net of exact duplicate rows ≥ 0.98) with coverage ≥ 0.8 (row-weighted when the full distribution is known) | 0.7·key score + 0.3·coverage |
| ENTITY_KEY_MERGE | both sides unique on the same id space, Jaccard ≥ 0.3 | 0.7·key score + 0.3·Jaccard |
| ENTITY_RESOLUTION | ≥ 2 descriptive matches incl. a discriminative one (name/e-mail/phone), comparable table sizes | mean top-3 × (0.95 or 0.85) |
| AGGREGATE_LOOKUP | shared non-unique categorical key with coverage ≥ 0.8 | 0.85 × (0.7·score + 0.3·coverage) |

The most reliable candidate wins; the structural priority only breaks near-ties (within 0.05).

## 2a. Value crosswalks (`backend/matching/crosswalk.py`)

Candidate pairs: text columns from different datasets with complete sketches (3–400 values, smaller side ≤ 80), not identifier/contact/date/number types, not already accepted, value containment < 0.5; the 6 with the highest first-pass score. The LLM receives both distinct-value lists and returns `{value_A: value_B | null}` (cached by the value lists). Verification: ≤ 20% of proposed targets absent from column B, ≥ 2 non-trivial pairs, coverage of A ≥ 0.5, injectivity ≥ 0.8. An accepted mapping is registered as runtime normaliser `crosswalk:<id>` for that column pair only, and matching runs a second pass in which value overlap and pattern similarity are measured after mapping. Results: 3/3 planted vocabulary correspondences found (0/3 without), negative control rejected.

## 3a. Relationship robustness rules (`profiling/semantic_types.py`, `matching/matcher.py`, `matching/value_sim.py`, `graph/verification.py`, `graph/relationships.py`)

* **Token identifiers:** a value signature (single token, letters and digits, 12–64 chars, near-constant length) types hashes and UUID-like ids as `ID`, including low-uniqueness foreign keys.
* **Identifier veto:** two `ID` columns sharing < 5% of values are excluded before the 1:1 assignment.
* **Prefixed codes:** the `identifier` normaliser is not used when both columns are coded with disjoint prefixes.
* **Row agreement:** attribute pairs of key-joined tables are compared on ≤ 20K sampled joined rows with the match's normaliser (numeric tolerance, shared timestamp precision); agreement < 0.5 rejects the pair.
* **Uniqueness rules:** 1:1 entity merges need both keys ≥ 0.999 unique; when both sides pass the 0.98 key threshold, the less unique side is the reference.
* **Sibling removal:** an X—Y link is dropped when both columns reference the same strictly unique key elsewhere (fan-trap avoidance).
* **Chance overlap of integer ranges:** when the key is dense on [l, h] and a reference's containment is no more than the share of its own range inside [l, h] (+0.05), the overlap is uninformative. The link then needs name support: a shared content word (camelCase/snake split, singular; *id, key, code, no* ignored) with the key column or with the key's table. This applies to lookups, key merges and aggregated lookups.
* **Alternate keys vs roles:** references from several fact columns to one key column are roles; references to several different columns of one dimension are one link, keyed on the best identifier (ID type, score, uniqueness), and the rest become attribute matches.
* **Shared attributes:** an aggregated lookup on a column unique on neither side needs one table to be aggregatable, i.e. keyless or at least half quantities.
* **Record linkage gate:** at least one discriminative field (name, e-mail, phone) must share ≥ 2% of values across the two tables.
* **Schema map layout:** directed relationship graph → cycle breaking at the weakest edge → longest-path layering → barycenter ordering, one block per component.

## 4. Graph algorithms (`backend/graph/algorithms.py`)

* **Connected components**: integration ecosystems under a confidence threshold.
* **Maximum spanning tree** (Kruskal): the join structure; excluded edges are reported with the stronger replacing path and its reliability.
* **Dijkstra** with cost `−log c` (or `1 − c`): best integration route; **Yen's k-shortest simple paths** for alternatives.
* **Column route**: Dijkstra over the column graph.
* **Column groups**: connected components over accepted matches, which give the canonical attributes.

## 5. Entity resolution (`backend/entity_resolution/`)

* **Blocking**: standard keys (e-mail, phone digits, identifiers), token key `surname|first-initial`, sorted neighbourhood (window 5) on `surname firstname`; blocks > 2 000 are skipped and counted. **Adaptive refinement:** a token block larger than `refine_block_size` (150) is split by the normalised city/country/category/zip value, then by the full first name if still too large (records without the value form their own sub-block). At 40K entities: 6.5M → 1.8M pairs, 189 → 54 s, 7.5 → 2.5 GB, F1 0.986 → 0.981.
* **Comparators** return discrete levels. Person names use token logic rather than whole-string Jaro–Winkler, which scores "Rahul Sharma" vs "Rohan Sharma" ≈ 0.9.
* **Fellegi–Sunter** with u from 4 000 random pairs and m from **rule-blocked EM**: for each strong field r, EM runs on pairs agreeing on r using only the other fields, and m values are averaged over the runs where a field was not the rule. Agreement levels are constrained never to count as evidence against a match. λ is re-estimated with m, u fixed. Identical comparison vectors are grouped, so EM is O(patterns).
* **Correlation clustering (GAEC)**: edge weight `logit(p) − logit(t)`, uncompared pairs −0.4. The cluster pair with the largest positive total weight is contracted repeatedly until none remains.
* Optional 1:1 linkage by maximum-weight matching (`one_to_one_links`).
* **Calibrated prior.** After rule-blocked EM, λ is re-estimated as the fixed point of λ = (1/T)·Σ_candidates σ₂(logit₂ λ + LLR) over all T comparable pairs (`prior_over_all_pairs`), so λ and u describe the same population. Merge modes carry record-link thresholds (strict 0.90 / balanced 0.50 / permissive 0.35) separate from schema-level thresholds. Optional Winkler term-frequency u for exact agreements (`term_frequency`, off: no gain measured).
* **Active learning.** `review_queue` ranks compared pairs by margin to the auto-merge threshold, `exp(−|logit p − logit t|)`, plus 0.5 when the clustering disagrees with the pairwise decision. The session queue adds 30% uniformly random pairs. Labels become must-/cannot-link constraints (±10⁶ in GAEC). Field-dependence check (log-linear interaction terms): lift P(both exact | M) / (P(j exact | M)·P(k exact | M)) is 0.996–1.000 for all field pairs on sparse data, so interactions were not added. Only randomly sampled labels recalibrate the match weights, via semi-supervised Platt scaling `σ(a·W + b)` fitted on all pairs' soft targets plus the labels (weight 20). Calibrating on uncertainty-sampled labels collapsed recall in the experiment, because boundary pairs are a biased sample.

## 6. Merge planning and execution (`backend/merge/`)

* **Root** = the dataset that keeps the most datasets un-aggregated: for each candidate, count the datasets reachable through same-entity joins and lookups whose fact side is the parent. Ties go to a dataset that is never a lookup dimension, then to the larger one; a pinned primary key overrides. On a star schema this is the fact table. The previous rule (largest non-dimension) rooted Olist at 1M geolocation points.
* **Join statistics** use each attachment's own match flag; flags of nested joins are carried as ordinary child columns.
* **Orientation**: a lookup whose fact side is the child becomes `reverse_aggregate`; ER/key-merge edges become entity groups anchored at the member closest to the root.
* **Roles**: `role_for(PULocationID, LocationID) = pickup` (tokens of the reference minus tokens of the key, ≤ 2 tokens).
* **Projection**: a static tree of column specs with internal view names; identical subtrees (the zone table under both roles) share one spec and are computed once.
* **Execution**: dataset views apply transformations in SQL; keys are normalised in SQL (basic, identifier) or via a distinct-value mapping table (Python normalisers); lookups deduplicate the child with `QUALIFY row_number()`; aggregations use `CASE WHEN count(DISTINCT x) <= 1 THEN min(x) ELSE agg(x) END` so functionally dependent values pass through; entity fusion loads group records, resolves, and fuses per attribute with conflict capture.
* **Name-variant unification**: fused names that are pairwise name-compatible are treated as representation variants, not conflicts.

* **Attribute history** (`merge/temporal.py`): for descriptive attributes with ≥ 2 distinct dated values, candidates are sorted by change timestamp (creation timestamp as fallback, labelled). Ties go to source priority. Runs of equal normalised values become intervals `[valid_from, valid_to)`, and the last one is current. Validation checks the intervals are non-empty and non-overlapping.
* **Row uncertainty** (`_probability_terms`): `_match_probability` = product of the fused-entity cluster confidences on the row's lineage (record-level, ≈ independent across rows). `_relationship_confidence` = product of the confidences of matched lookups, followed through nested joins (schema-level, shared by all rows using the relationship).

## 7. Conflict resolution (`backend/conflicts/strategies.py`)

`majority_vote` (default) · `prefer_source` · `prefer_latest` (uses a detected record-timestamp column, else majority) · `prefer_non_null` · `highest_confidence` · `source_accuracy_vote` (iterative truth discovery with Laplace-smoothed per-source accuracy, votes weighted by `log(a/(1−a))`) · `keep_all` · `manual_review`. Every conflicting cell is recorded with all candidates, the chosen value and the reason.

## 7a. Learned column matcher (`backend/matching/learned.py`)

20 pair features (7 signals, Jaccard, two-way containment, graph support, exclusivity, type compatibility, uniqueness, distinct ratio, type flags) → `HistGradientBoostingClassifier` (balanced, l2 = 1) wrapped in isotonic `CalibratedClassifierCV`. Trained by `experiments/train_matcher.py` on 36 fabricated scenarios (seed 7777; the evaluation uses 2026), with GroupKFold by scenario for the out-of-fold AUC (0.9998) and Brier (0.0021). `schema_matching.scorer: weighted | learned | blend` (blend = mean of weighted score and probability). `Session.retrain_matcher` appends approve/reject decisions with weight 10. The matcher is not the default: NYC F1 is 0.400 learned vs 0.769 weighted.

## 7b. Uncertain aggregates (`backend/provenance/uncertainty.py`)

Rows ~ independent Bernoulli(`_match_probability`). Per group: point Σx, expected Σp·x, variance Σp(1−p)x², normal 95% interval, certain-only Σ_{p≥τ} x, min relationship confidence. Count uses x = 1; avg uses Σp·x / Σp. One DuckDB aggregation over the Parquet output.

**Scenarios.** The executor records every matched join with its chain from the root, confidence and marker column (`_src_<alias>_row` for top-level joins, a match flag for nested ones). For each join below the high-confidence threshold whose chain prefixes the lineage path of the measure or group column, the aggregate is recomputed without the rows joined through it and reported with probability 1 − confidence.

**Bitemporal history.** History rows carry `recorded_at` (merge time). `as_known_at` selects the latest merge recorded at or before the given time, including undone and outdated merges.

## 8. Conversational agent (`backend/llm/`)

* **Tool calling.** 28 tools with JSON schemas. The model only chooses tools and arguments; every argument is validated (types, enums, ranges, unknown keys rejected) and executed by a `Session` method. All figures in replies come from deterministic renderers.
* **Agent loop.** Up to 5 rounds of *model → tool calls → validated execution → tool results back to the model*, then at most 3 sentences of commentary. The system prompt carries a live session summary, so users never repeat dataset names.
* **Schema compaction.** Tool schemas are sent without JSON noise, and at most 12 of the 28 are offered per request (§8a); all 28 would be ≈ 2.5K tokens. Optional parameters are declared nullable: models often send omitted arguments as `null`, and providers that validate tool calls server-side (Groq) otherwise reject the call.
* **Rate-limit pacing.** After each response the provider records `x-ratelimit-remaining-tokens` and the reset time. Before the next request it estimates the request size ($`\approx |\text{body}| / 4`$ tokens plus a quarter of `max_tokens`) and waits for the window when the request would not fit. HTTP 429 honours `Retry-After`, otherwise backs off exponentially ($`2 \cdot 2^{a}`$ s, capped at 30 s).
* **Failure handling.** A total latency budget applies per call. If the model fails after tools already ran, those results are returned and nothing is re-executed. If it fails before any tool ran, the deterministic rule-based parser handles the turn.
* **Rule-based parser.** Multi-intent regex grammar over session-aware dataset/column mentions. Preference phrases are consumed before action detection. Offline and reproducible; used by the tests and as the fallback.

## 8a. Rate-limit control (`backend/llm/rate_limit.py`, `tool_selection.py`)

* **Limiter.** Per model it keeps sliding windows: (timestamp, tokens) for 60 s, and per-minute buckets for 24 h. Before a request it estimates `chars/3.2 + expected output`. A day-budget overrun raises at once; a minute-budget overrun sleeps until enough of the window expires (≤ `max_wait_seconds`) or raises. After the request, reported usage replaces the estimate, and `x-ratelimit-remaining-*` headers tighten the view. A 429 body is parsed for scope and retry time and blocks the model. State is persisted to JSON.
* **Out-of-subset recovery.** A server-side 400 for a tool that was not offered triggers one retry of that round with all tools; in follow-up rounds the agent stops and keeps the executed results. Unknown argument names are dropped (logged), never passed on.
* **Failover.** Models are tried in order: the primary may wait, the fallbacks may not. If all are minute-limited, the soonest slot is awaited once (≤ 75 s).
* **Tool selection.** It takes the union of a core set (by session state), rule-parser intents, column-mention triggers (≥ 2 session columns → `explain_match`, `decide_mapping`) and keyword overlap with tool name, description and synonyms, up to 12 tools. Follow-up rounds offer mutating tools only when the current message requests a change (parser intent or action words).

## References

Fernandez et al., *Aurum*, ICDE 2018 · Bogatu et al., *D3L*, ICDE 2020 · Koutras et al., *Valentine*, ICDE 2021 · Melnik, Garcia-Molina, Rahm, *Similarity Flooding*, ICDE 2002 · Madhavan, Bernstein, Rahm, *Cupid*, VLDB 2001 · Zhu et al., *LSH Ensemble*, VLDB 2016; *JOSIE*, SIGMOD 2019 · Fellegi & Sunter, JASA 1969 · Hernández & Stolfo, *Merge/Purge*, SIGMOD 1995 · Hassanzadeh et al., *Framework for evaluating clustering algorithms in duplicate detection*, VLDB 2009 · Keuper et al., *Efficient decomposition of image and mesh graphs by lifted multicuts* (GAEC), ICCV 2015 · Bleiholder & Naumann, *Data Fusion*, ACM CSUR 2008 · Dong, Berti-Equille, Srivastava, *Integrating conflicting data*, VLDB 2009 · W3C PROV-O, 2013.
* Platt, *Probabilistic outputs for support vector machines*, 1999; Zadrozny & Elkan, *Transforming classifier scores into accurate multiclass probability estimates*, KDD 2002.
* Sarawagi & Bhamidipaty, *Interactive deduplication using active learning*, KDD 2002.
* Dalvi & Suciu, *Efficient query evaluation on probabilistic databases*, VLDB 2004.
* Snodgrass (ed.), *The TSQL2 Temporal Query Language*, 1995.
