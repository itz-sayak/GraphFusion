# API reference

Base URL `http://localhost:8000`; interactive docs at `/docs` (OpenAPI).

**Sessions.** Send `X-Session-ID: <id>` (or `?session_id=<id>`). An unknown or missing id creates a session, and the id is returned in the `X-Session-ID` response header.

**Errors.**

```json
{"error": "not_found", "message": "Unknown dataset 'x'", "details": {"available": ["customers", "sales"]}}
```

| Status | `error` |
|---|---|
| 400 | `dfg_error`, `unsupported_format`, `ingestion_failed` |
| 404 | `not_found` |
| 409 | `invalid_state` (e.g. export before any merge) |
| 422 | `invalid_request`, `invalid_tool_arguments`, `validation_failed` |
| 502 | `llm_provider_error` |

## Meta and sessions

| | |
|---|---|
| `GET /health` | status, version, active LLM provider |
| `GET /config` | matching weights, thresholds, merge modes, conflict strategies |
| `POST /sessions` · `GET /sessions` · `GET /sessions/current` · `DELETE /sessions/{id}` | session management; `GET /sessions` includes sessions persisted on disk; delete removes the workspace and saved state |

**Persistence.** After every successful POST/PUT/PATCH/DELETE the session is saved to `sessions/<id>/session_state.pkl`; the first request after a restart with the same `X-Session-ID` restores it.

**Authentication (optional).** With `DFG_API_TOKENS=alice:tok-a,bob:tok-b`, every path except `/health`, `/docs`, `/redoc` and `/openapi.json` requires `Authorization: Bearer <token>` (or `?token=<token>`, used by download links); otherwise `401 {"error":"unauthorized"}`. Sessions belong to the user who created them: other users get 404 and do not see them in `GET /sessions`.

`GET /llm/usage`: per-model rate-limit accounting (requests and tokens in the last minute and 24 h, configured limits, blocked-until).

Quick checks: `curl http://localhost:8000/health` · `curl http://localhost:8000/llm/usage` · `curl -o /dev/null -w "%{http_code}" http://localhost:8000/docs`. A full verified command walkthrough (curl and PowerShell) is in README §12.

## Datasets

```bash
curl -F files=@customers.csv -F files=@sales.parquet http://localhost:8000/datasets/upload
curl -X POST -H 'Content-Type: application/json' -d '{"paths":["sample/sample_customers.csv"]}' http://localhost:8000/datasets/load-path
curl -X POST http://localhost:8000/demo/nyc
```

| | |
|---|---|
| `POST /datasets/upload` | multipart `files`; returns loaded summaries and per-file errors. A SQLite file with several tables loads one dataset per table; a CSV the sniffer cannot split is repaired (`metadata.csv_repair`) |
| `POST /datasets/load-path` | `{"paths": [...]}` relative to `data/` (traversal rejected) |
| `POST /datasets/load-url` | `{"url", "name", "params"?, "page_size"?}` JSON REST source (optional) |
| `POST /demo/{sample\|nyc}` | load a demo scenario |
| `GET /datasets` · `GET /datasets/{id}` · `DELETE /datasets/{id}` | list / detail / remove from session (source untouched) |
| `GET /datasets/{id}/profile?refresh=false` | `DatasetProfile` |
| `GET /datasets/{id}/preview?limit=20&offset=0` | rows |

## Integration

| | |
|---|---|
| `POST /integration/discover` `{"force": false}` | datasets, relationships, mappings, rejected candidates, components, issues, graph stats |
| `GET /integration/graph?level=schema` | schema map: `tables` (columns with role key/reference/linked/attribute, uniqueness, nulls, glossary description, layer/order/component), `links` (column → column, kind join_key/attribute/candidate/rejected, relationship, confidence, row_agreement, in_merge_tree), `relationships` (cardinality, key pairs), `stats` |
| `GET /integration/graph?level=dataset\|column` | older React-Flow nodes/edges with evidence |
| `GET /integration/graph/full` | full `integration_graph.json` |
| `GET /integration/mappings` | every scored correspondence with factor breakdown, dataset relationships, canonical schema |
| `POST /integration/mappings/decision` `{"left","right","decision":"approved\|rejected"}` | user override |
| `GET /integration/explain/match?left=city&right=location` | factors, weights, normaliser, graph adjustments, decision |
| `GET /integration/explain/relationship?left=A&right=B` | join kind, keys, coverage, or route + rejected candidates |
| `GET /integration/route?source=A&target=B&cost_mode=neglog` | Dijkstra path, hops, reliability, alternatives |
| `POST /integration/entities/preview` `{"left","right"?,"limit"}` | ER statistics, Fellegi–Sunter parameters, confident / ambiguous pairs |
| `GET/POST /integration/preferences` | `mode`, `conflict_strategy`, `confidence_threshold` (0–1 or %), `clear_threshold`, `primary_key`, `merge_uncertain`, `source_priority` |
| `POST /integration/plan` `{"mode"?, "conflict_strategy"?, "datasets"?}` · `GET /integration/plan` | `MergePlan` |
| `POST /integration/execute` `{"plan_id"?, "write_csv": true}` | `MergeResult` + plan, files, join stats, entity-group stats |
| `POST /integration/undo` | archive the latest merge |
| `POST /integration/matcher/retrain` | refit the learned column matcher with this session's mapping decisions (409 if none) |
| `GET /integration/entities/review?limit=20` | pairs to label: `left`/`right` (`dataset`, `row`, `source`, `values` by field), `probability`, `sampling` (`uncertainty`\|`random`), `same_entity_now`, `field_levels` |
| `POST /integration/entities/label` `{"left":{"dataset","row"},"right":{…},"decision":"match\|non_match","sampling"?}` | store a label; applied as a constraint on the next merge (random-sample labels also recalibrate) |

## Results

| | |
|---|---|
| `GET /integration/output?limit&offset` | unified dataset rows |
| `GET /integration/conflicts?limit&attribute&entity_id` | conflicts with candidates |
| `GET /integration/duplicates` | exact duplicate rows, within-dataset duplicates, cross-source links |
| `GET /integration/provenance?column=` | column lineage (all or one) |
| `GET /integration/lineage-report` | text report |
| `GET /integration/quality` | quality report, validation checks, match statistics |
| `GET /integration/history?entity_id&attribute&as_of&as_known_at&limit` | validity intervals (`value`, `valid_from`, `valid_to`, `is_current`, `timestamp_kind`, source, `recorded_at`); `as_of` (ISO, valid time) returns values valid then; `as_known_at` (ISO, transaction time) reads the history of the latest merge executed at or before that time (409 if none) |
| `GET /integration/aggregate?measure&group_by&agg=sum\|count\|avg&limit` | per group: `point`, `expected`, `ci95_low`/`ci95_high`, `certain_only`, `mean_probability`, `min_relationship_confidence`; plus `expected_correct_rows` |
| `GET /integration/export` · `GET /integration/export?file=unified_dataset.parquet` | list / download |

## Chat

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"message":"Only merge matches above 95%, then merge them","session_id":"<id>"}' \
  http://localhost:8000/chat
```

Response:

```json
{
  "session_id": "…",
  "reply": "markdown",
  "tool_calls": [{"tool": "set_preferences", "arguments": {"confidence_threshold": 0.95}, "ok": true, "error": null}],
  "cards": [{"type": "merge", "tool": "execute_merge", "data": {}}],
  "suggestions": ["Show me conflicts", "Export the final dataset"],
  "provider": {"provider": "groq", "model": "qwen/qwen3.8-27b", "mock": false, "fallback_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]},
  "elapsed_ms": 8421.3
}
```

`provider` may be overridden per request (`"groq"`, `"nvidia_nim"`, `"xai_grok"`, `"openai_compat"`, `"anthropic"`, `"ollama"`, or `"mock"` for the offline rule-based parser). If the model fails or exceeds `DFG_LLM_TIMEOUT`, `provider.fallback` is `true`. `GET /chat/history` returns the conversation.
