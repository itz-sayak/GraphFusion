# Performance benchmark (NYC integration at scale)

Machine: {'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'logical_cpus': 28, 'ram_gb': 16.9, 'os': 'Windows-10-10.0.26200-SP0', 'python': '3.11.9'}

| trip rows | DuckDB memory limit | ingest (s) | profile (s) | match+graph (s) | plan (s) | merge+validate+export (s) | total (s) | peak RSS (MB) | rows/s |
|---|---|---|---|---|---|---|---|---|---|
| 100,000 | 2GB | 0.33 | 1.7 | 2.64 | 0.0 | 0.55 | 5.22 | 457.3 | 19,155 |
| 500,000 | 2GB | 0.35 | 2.66 | 2.67 | 0.0 | 0.93 | 6.61 | 1016.9 | 75,615 |
| 1,000,000 | 2GB | 0.37 | 3.79 | 2.61 | 0.0 | 1.51 | 8.28 | 1488.0 | 120,739 |
| 1,000,000 | 512MB | 0.37 | 4.0 | 2.65 | 0.0 | 3.01 | 10.03 | 899.3 | 99,667 |

Profiling and schema matching hold only bounded samples and sketches, so their memory is flat in the number of rows. Merge execution memory is governed by `engine.memory_limit`: intermediates are lazy DuckDB views and the result is streamed to Parquet, so DuckDB uses memory up to the limit and spills to disk beyond it. The last row repeats the largest size with a 512MB limit: lower peak memory in exchange for a slower merge stage. Peak RSS includes the Python process baseline (~250-300MB).