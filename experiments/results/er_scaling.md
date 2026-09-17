# Entity resolution scaling (single process, adaptive block refinement on)

| entities | records | candidate pairs | seconds | records/s | peak RSS (MB) | pairwise F1 | blocking (ms) | scoring (ms) | clustering (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 3337 | 22083 | 0.8 | 4057 | 210 | 0.993 | 64 | 608 | 103 |
| 10000 | 16569 | 416738 | 11.0 | 1504 | 707 | 0.988 | 254 | 8598 | 1834 |
| 40000 | 66028 | 1791335 | 53.8 | 1228 | 2505 | 0.981 | 1546 | 40254 | 10700 |
| 100000 | 164985 | 5109474 | 170.3 | 969 | 7026 | 0.978 | 4419 | 125338 | 35224 |