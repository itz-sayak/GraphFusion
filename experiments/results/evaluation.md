# Entity resolution and integration evaluation

Seeds: [7, 11, 23] (600 entities, ~1,000 customer records per seed).

## Entity resolution (pairwise, mean over seeds)

| method | precision | recall | F1 | time (s) |
|---|---|---|---|---|
| Exact e-mail | 1.000 | 0.872 | 0.932 | 0.00 |
| Fuzzy name+city | 0.795 | 0.822 | 0.808 | 0.01 |
| FS old prior + correlation (t=0.9) | 0.999 | 0.997 | 0.998 | 0.17 |
| FS + components (t=0.9) | 1.000 | 0.978 | 0.989 | 0.15 |
| FS + correlation (t=0.9) | 1.000 | 0.977 | 0.988 | 0.14 |
| FS + components (t=0.5) | 0.996 | 0.998 | 0.997 | 0.15 |
| FS + correlation (t=0.5) | 0.999 | 0.996 | 0.997 | 0.14 |

## End-to-end integration (mean over seeds)

| metric | DataFusionGraph | naive exact-name join |
|---|---|---|
| match_coverage | 0.8933 | 0.0 |
| row_match_rate | 0.9792 | 0.0 |
| unmatched_ratio | 0.0208 | 1.0 |
| conflict_rate | 0.0125 | n/a |
| duplicate_reduction | 1.0 | 0.0 |
| integration_accuracy | 0.9995 | 0.0 |
| true_orphan_rate | 0.0208 | 0.0208 |
| runtime_s | 2.0633 | — |
| peak_rss_mb | 274.8333 | — |

Naive baseline: shared column names between sales and customers: none — without schema matching nothing can be joined.

Notes: `row_match_rate` is bounded by the true orphan rate (transactions referencing customers that do not exist). `conflict_rate` = conflicting fused attribute values / fused attribute cells.