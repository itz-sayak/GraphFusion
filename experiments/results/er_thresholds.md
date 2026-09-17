# Record-link thresholds: old vs calibrated prior

Pairwise P/R/F1 (correlation clustering), mean of 3 seeds. Thresholds were chosen on the tuning seeds; the held-out seeds were not used for any choice.

| scenario | seeds | setting | record-link threshold | precision | recall | pairwise F1 |
|---|---|---|---|---|---|---|
| default | tuning (7, 11, 23) | old prior, old balanced | 0.90 | 0.999 | 0.997 | 0.998 |
| default | tuning (7, 11, 23) | calibrated, strict | 0.90 | 1.000 | 0.977 | 0.988 |
| default | tuning (7, 11, 23) | calibrated, balanced | 0.50 | 0.999 | 0.996 | 0.997 |
| default | tuning (7, 11, 23) | calibrated, permissive | 0.35 | 0.999 | 0.997 | 0.998 |
| default | held-out (101, 202, 303) | old prior, old balanced | 0.90 | 0.999 | 0.999 | 0.999 |
| default | held-out (101, 202, 303) | calibrated, strict | 0.90 | 1.000 | 0.979 | 0.989 |
| default | held-out (101, 202, 303) | calibrated, balanced | 0.50 | 0.999 | 0.999 | 0.999 |
| default | held-out (101, 202, 303) | calibrated, permissive | 0.35 | 0.999 | 0.999 | 0.999 |
| hard | tuning (7, 11, 23) | old prior, old balanced | 0.90 | 0.777 | 0.959 | 0.856 |
| hard | tuning (7, 11, 23) | calibrated, strict | 0.90 | 1.000 | 0.326 | 0.490 |
| hard | tuning (7, 11, 23) | calibrated, balanced | 0.50 | 0.922 | 0.898 | 0.909 |
| hard | tuning (7, 11, 23) | calibrated, permissive | 0.35 | 0.915 | 0.934 | 0.924 |
| hard | held-out (101, 202, 303) | old prior, old balanced | 0.90 | 0.834 | 0.968 | 0.895 |
| hard | held-out (101, 202, 303) | calibrated, strict | 0.90 | 1.000 | 0.358 | 0.526 |
| hard | held-out (101, 202, 303) | calibrated, balanced | 0.50 | 0.936 | 0.948 | 0.942 |
| hard | held-out (101, 202, 303) | calibrated, permissive | 0.35 | 0.932 | 0.949 | 0.940 |