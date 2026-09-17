# Entity resolution scaling (single process, adaptive block refinement off)

| entities | records | candidate pairs | seconds | records/s | peak RSS (MB) | pairwise F1 | blocking (ms) | scoring (ms) | clustering (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 3337 | 22083 | 0.8 | 4037 | 211 | 0.993 | 78 | 591 | 116 |
| 10000 | 16569 | 416738 | 8.4 | 1975 | 707 | 0.988 | 247 | 6413 | 1535 |
| 40000 | 66028 | 6512936 | 188.9 | 350 | 7517 | 0.986 | 2702 | 142827 | 38250 |