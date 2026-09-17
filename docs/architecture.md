# Architecture

## Layers

| Layer | Package | Responsibility |
|---|---|---|
| UI | `frontend/` | Chat, dataset sidebar (drag & drop), Graph / Schema / Mappings / Plan / Conflicts / Output / Provenance tabs |
| API | `backend/api/`, `backend/main.py` | REST + `/chat`, Pydantic models, structured errors, session header |
| Orchestration | `backend/session/state.py` | `Session`: holds all state and exposes every operation (tools call these methods) |
| Conversational | `backend/llm/` | `LLMProvider` abstraction (Groq default; NVIDIA NIM, xAI Grok, OpenAI-compatible, Anthropic, Ollama), tool registry and validation, agent loop, rate-limit pacing, rule-based fallback parser |
| Ingestion | `backend/ingestion/` | Format detection, lossless readers, immutable raw copies + SHA-256 |
| Profiling | `backend/profiling/` | DuckDB statistics, semantic types, coordinated bottom-k sketches, MinHash |
| Matching | `backend/matching/` | Name normalisation, 7 similarity signals, LSH candidates, similarity flooding, bipartite alignment |
| Entity resolution | `backend/entity_resolution/` | Blocking, comparators, Fellegi–Sunter + rule-blocked EM, correlation clustering |
| Graph | `backend/graph/` | Integration graph, relationship inference, algorithms, explanations, serialisation |
| Merge | `backend/merge/` | Modes, planner, static projection, transformations, DuckDB executor |
| Conflicts | `backend/conflicts/` | Fusion strategies, truth discovery |
| Provenance / validation | `backend/provenance/`, `backend/validation/` | Lineage, PROV-JSON, integrity checks, quality report |
| Storage | `backend/storage/` | Workspace layout, DuckDB helpers, export bundle |

## Data flow and state

```text
upload ─► Workspace/raw (read-only copy, sha256) ─► processed/<id>.parquet ─► profile + sketches
      ─► discover: candidates → scores → flooding → alignment → relationships → IntegrationGraph
      ─► plan: filter by mode → components → MST → root → orientation → groups → transforms → projection
      ─► execute: DuckDB views → lookups/aggregations → entity fusion (Python) → unified view
      ─► stream unified → Parquet ─► validate ─► provenance ─► export bundle ─► MergeRecord (undo stack)
```

Workspace layout (`DFG_WORKSPACE`, default `./workspace`):

```text
sessions/<session_id>/
  raw/            byte-identical, read-only copies of every source
  processed/      normalised Parquet per dataset
  merges/<id>/    unified_dataset.parquet|csv, reports, provenance, conflicts, entities, graph
  merges/_undone/ archived outputs of undone merges
  _duckdb_tmp/    DuckDB spill directory
  session.json    snapshot of session state (summary)
  session_state.pkl  full session (datasets, decisions, labels, crosswalks, merges, chat); written atomically
_llm_usage.json   rate-limiter windows shared by all sessions
_crosswalk_cache.json  LLM crosswalk proposals keyed by value lists
```

**Persistence and users.** Middleware in `backend/main.py` saves the session (in a worker thread) after each successful mutating request. `SessionRegistry` restores a session lazily from `session_state.pkl` on first access; the lock and runtime crosswalk normalisers are rebuilt in `__setstate__`. With `DFG_API_TOKENS`, the same middleware checks the bearer token and stores the user; sessions record `owner`, and the registry hides other users' sessions.

## Relationship checks in the discovery flow

```text
profile ─► match_schemas (token-ID typing, identifier overlap veto before alignment, prefix-aware identifier normaliser)
        ─► infer_relationships (strict 1:1, more-unique side referenced, no measure keys, sibling links removed)
        ─► verify_attribute_matches (row agreement on key-joined samples)
        ─► integration graph ─► schema map (GET /integration/graph?level=schema)
```

## Memory and scale

| Stage | Memory | Time |
|---|---|---|
| Ingestion | streaming (DuckDB `COPY … TO parquet`); CSV read twice (type validation pass + copy) | O(rows) |
| Profiling | aggregates in DuckDB; Python keeps ≤ 20 000 bottom-k values, 25 top values and a 1 000-row reservoir per column → O(columns · k) | O(rows · columns) |
| Schema matching | O(columns · k) sketches; candidate pairs exhaustive below 5 000 pairs, LSH Ensemble above | ~O(C log C) with LSH |
| Entity resolution | Python over entity-group records: O(records + candidate pairs); token blocks > 150 refined by city/category then first name, blocks capped at 2 000 | O(Σ block²); 165K records in 170 s / 7 GB |
| Merge | lazy DuckDB views; hash joins on dimension tables; result streamed to Parquet; spills beyond `engine.memory_limit` | O(rows) |
| Validation / preview | queries on the written Parquet (columnar, projection pushdown) | O(rows) scans |

Fact tables (millions of rows) never enter Python. Dimension and entity tables do, during fusion only.

Measured (see `experiments/results/benchmark.md`): 1M trips end to end in 8.3 s. Peak RSS is 1.5 GB with a 2 GB DuckDB limit and 0.9 GB with a 512 MB limit.

## Integrity guarantees

* Sources are never opened for writing. Raw copies are read-only and checksummed, and validation re-verifies the checksums.
* Lookups deduplicate the child on its key first, so the parent's row count is invariant (checked: `row_conservation`, `no_fan_out`, `join_row_invariance`).
* Unmatched rows are kept with `_match_* = false`, never dropped.
* Conflicting values are recorded with all candidates, whatever strategy resolves them.
* Every transformation keeps the raw value (`*_raw`).
* Undo archives outputs instead of deleting them.
* Attribute history intervals are validated after every merge (`temporal_consistency`).
* Every output row carries `_match_probability` (record-level) and, when lookups are used, `_relationship_confidence` (schema-level); the quality report states the expected number of correctly integrated rows.
* User labels (column decisions, entity-link labels) are stored in the session and applied on the next discovery/merge; they never modify sources.

## Conversational layer

```text
/chat ─► ChatAgent ─► provider (default: Groq · qwen/qwen3.8-27b → gpt-oss-120b → gpt-oss-20b)
            │            ├─ ≤ 12 relevance-selected tool schemas (of 28; optional args nullable)
            │            ├─ predictive per-model limiter (RPM/RPD/TPM/TPD, persisted) + failover on 429
            │            ├─ live session summary in the system prompt
            │            └─ pacing from x-ratelimit-remaining-tokens / reset-tokens; Retry-After on 429
            ├─ validate arguments ─► Session method ─► deterministic renderer
            └─ on failure: partial results (never re-run tools) or rule-based fallback parser
```

The provider comes from `DFG_LLM_PROVIDER`. The rule-based parser (`mock`) needs no network. It serves as the automatic fallback, and the test suite uses it so tests are reproducible offline.

## Observability

* `GET /llm/usage`: per-model request/token counts for the last minute and 24 h, limits and blocks (`workspace/_llm_usage.json`).

`structlog` events (console or JSON with `DFG_LOG_JSON=true`): `Dataset ingested`, `Schema profiling completed`, `Candidate relationships discovered`, `Graph constructed`, `Merge plan generated`, `Join executed`, `Entity resolution completed`, `Entity fusion completed`, `Conflicting field detected` (warning), `Merge completed`, `Merge exported`, `LLM call`, `Pacing for provider rate limit`, `Tool executed`, `HTTP request`.
