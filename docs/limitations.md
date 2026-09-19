# Limitations

## Matching

* **Correspondences without evidence in values or names are missed.** `cust_no` (`C-50123`) and `customer_id` (`1024`) are the same attribute in different id systems. With zero value overlap they score about 0.52 and stay below the threshold, so recall on the sample scenario is capped at 0.714. The system links those entities through descriptive attributes instead.
* **Opaque codes** (`B19013_001E`) are matched on values alone unless the business glossary maps them. The glossary is maintained by hand.
* **Reference value maps are configuration.** City aliases, borough ↔ county and country synonyms must exist in `config/value_maps.yaml`; unseen aliases are not bridged. There is no learned value-embedding matcher.
* **Similarity flooding adds little.** In the ablation most of the graph benefit comes from bipartite alignment; flooding adds about 1 F1 point on medium/hard scenarios and loses 2 on easy ones at the default threshold. It is kept because it improves precision on harder cases and makes the transitive evidence visible, but it is not the main reason the system works.
* **Embeddings default to character n-gram TF-IDF**, which captures morphology rather than meaning (`earnings` vs `income` needs the thesaurus or the optional Sentence-Transformers backend).
* Scores are calibrated heuristically (weights in config), not learned. Thresholds transfer between the scenarios tested but are not guaranteed to transfer everywhere.

## Entity resolution

* It runs in one Python process over the records of entity groups. With adaptive block refinement 165K records take 170 s and 7 GB; about 200K records is the practical ceiling on 17 GB. It is not a distributed ER system.
* Adaptive refinement loses name-only matches whose city/category differs (moved customers without a shared e-mail or phone): −0.005 F1 at 40K entities.
* The mid-band under-confidence is not caused by field dependence (lift ≈ 1.0); the cause is open.
* Blocking can miss matches whose names, e-mails and phones all disagree at once.
* Clustering is a greedy heuristic for an NP-hard objective.
* EM needs enough agreeing pairs per rule (≥ 20). On tiny tables priors dominate.
* **Probabilities are conservative when identifying fields are sparse.** After re-estimating the prior over all pairs (pair ECE 0.093 → 0.016), pairs predicted 0.5–0.9 are true matches 0.1–0.2 more often than predicted, so expected counts from `aggregate_with_uncertainty` under-state correct rows. Two-parameter recalibration from 30–60 labels (uniform or stratified) did not fix the mid-band bias.
* Record-link thresholds are per mode (strict 0.90 / balanced 0.50 / permissive 0.35) on calibrated probabilities; strict mode keeps precision at 1.0 but recall falls to about 0.33 on sparse data.
* Pretrained sentence encoders (MiniLM, bge-small) did not improve schema matching on real data and slightly hurt the hard fabricated set; the misses that remain need world knowledge or crosswalks.
* Active learning: with the old overconfident probabilities, calibrating on uncertainty-sampled labels collapsed F1 (0.711); with calibrated probabilities it no longer does (0.957), and the hybrid default is within 0.005 of the best variant.
* Relationship (schema-level) confidence is reported as scenarios (one uncertain relationship removed at a time), not propagated into intervals; combinations of several wrong relationships are not enumerated.

## Relationship inference

* Only single-column keys are detected; composite keys are not.
* Small-integer references named after neither the key nor its table are missed (Northwind `shipVia`, Chinook `SupportRepId`): overlap of dense 1..n ranges is treated as no evidence.
* Probabilistic links between tables of the same kind of entity can still appear when a few identifying values recur by coincidence (Northwind shippers ~ suppliers via one phone number; Chinook Genre ~ Playlist via shared names). Records are only fused above the record-link threshold.
* Tables whose only common evidence is abbreviated or misspelt names are not linked directly (record linkage needs ≥ 2% exactly recurring identifying values); a route through a third table still works.
* Row-level verification of attribute correspondences needs a key join (lookup or 1:1); aggregated lookups and probabilistic entity links are not verified row by row, and pairs with < 30 comparable joined rows stay unverified.
* The identifier veto rejects genuinely related identifier systems that share no values (e.g. an old and a new customer numbering without a crosswalk); linking those needs a mapping table.
* The robustness generator (`experiments/run_schema_robustness.py`) was written alongside the rules and uses one star-schema shape; it guards known failure modes rather than proving general correctness.

## Merging

* Two files with the same schema (e.g. monthly exports) are treated as related datasets and joined, not appended; a union step is not implemented.

* One integration tree per plan: datasets outside the main connected component are reported and left unmerged, not merged separately.
* With several fact tables (Chinook invoice lines and playlist tracks) one grain is chosen, the one keeping more tables un-aggregated; the other fact table is aggregated onto it. Pin `primary_key` for the other grain.
* CSV repair of rows with unquoted delimiters is heuristic (type consistency plus the ", " cue); ambiguous choices are counted in the dataset metadata, and the original file is kept.
* Discovery on 9 Olist tables (1.55M rows) took 92 s; candidate generation and scoring grow with the number of table pairs.
* Aggregation functions are chosen from names and types (sum for counts/amounts, avg for rates/medians). Values constant within a group pass through, but a genuinely non-additive measure with a count-like name could be summed wrongly. The `_sum`/`_avg` suffix makes the choice visible.
* Currency conversion uses a static, dated rate table rather than historical rates at transaction time.
* Attribute history models valid time only (no transaction time). When a source has only a creation date, intervals start at creation (`timestamp_kind = creation`), which is weaker than a change log. History is emitted only for descriptive attributes with at least two distinct dated values.
* Lookups pick the first row when a dimension key is duplicated after normalisation, with a warning; they do not attempt ER inside the dimension.

## Learned matcher

* Trained only on fabricated scenarios. It overfits the generator (fabricated F1 ≈ 1.0, NYC 0.400 vs 0.769 weighted), so `weighted` remains the default. User decisions are up-weighted but few in number.

## Evaluation

* The fabricated benchmark is generated. Schema noise was produced independently of the matcher's dictionaries, but real-world transfer is only partly evidenced by the two real scenarios (sample customers, NYC).
* NYC ground truth was authored by hand, and borough ↔ county correspondences are a judgement call; the system does not find them (recall 0.714 there).
* Benchmarks come from one machine (28 logical CPUs, 17 GB RAM, Windows 11).

## System

* Sessions persist as a local pickle file per session: safe only for workspaces you created, not shared across several backend processes, and a file from an incompatible state version is ignored (not migrated).
* Authentication is optional static bearer tokens (`DFG_API_TOKENS`) with per-user session ownership: no login, expiry, roles or TLS.
* Value crosswalks need a real LLM provider and small vocabularies (≤ 80 / ≤ 400 values); a small vocabulary mapped into a much larger one (5 boroughs vs 62 counties) still scores low on containment.
* Transaction time is the merge time, not when a source system learned a fact.
* **LLM rate limits.** The default provider is Groq (`qwen/qwen3.8-27b`, falling back to `openai/gpt-oss-120b` and `openai/gpt-oss-20b`). Limits are per model: 30 RPM, 1K RPD, 8K TPM, 200K TPD, and RPM and TPD are not in any header. After tool selection and context trimming a turn costs 2.3–4.2K tokens (it was ~7.5K). That allows about 60 turns per model per day, but only about 2 per minute. When all models are minute-limited, the chat waits up to 75 s; when all are day-limited, it falls back to the rule-based parser.
* Tool routing is not fully deterministic even at temperature 0.2: in two live runs of the 12-request sample demo, one run answered 3 requests in text without a tool call and used the rule-parser fallback once (both runs passed end to end). A request for a tool outside the selected subset costs one extra full-tool round.
* The limiter's token estimate is heuristic, and Groq's day window is rolling and only reported through 429s, so a model can still receive one 429 before being blocked.
* Failover mid-conversation can move a turn to a smaller model (gpt-oss-20b), which chooses tools less reliably. The §17 LLM benchmark predates the model change and tool selection.
* Tool selection sends at most 12 of 28 schemas. Its 40/40 recall was measured on the benchmark requests its synonym list was written with, so real recall is likely lower.
* The LLM benchmark predates the 4 newest tools (`review_entity_links`, `label_entity_link`, `entity_history`, `aggregate_with_uncertainty`). A re-run after the nullable-argument fix got 12/12 correct before the free tier's daily token quota (200K) ran out.
* The LLM chose a wrong operation for 2 of 40 benchmark requests (5.0%), all without unrequested data actions (`experiments/results/llm_eval.md`). Mutating operations are validated but not confirmed interactively.
* The rule-based fallback parser gets only 0.444 of paraphrased requests right and executed unrequested merges in 2 of 40 cases.
* Remote LLM latency depends on the provider. During development NVIDIA NIM's free tier queued requests for minutes (`kimi-k3` produced the correct tool call after 4.5 min, and two flash models timed out at 200 s). The agent enforces a latency budget and falls back to the rule-based parser.
* The Docker images were written but not built on the development machine (Docker daemon unavailable).
