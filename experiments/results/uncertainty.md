# Calibration of match probabilities (probabilistic provenance)

Multi-record entities from the customer scenarios, seeds [7, 11, 23]. Predicted = cluster confidence (the `_match_probability` of the entity's output row); observed = purity (all records belong to one true entity). ECE uses 10 equal-mass bins. "95% interval covers" counts the seeds where the observed number of pure entities lies in Σp ± 1.96·√Σp(1−p).

| scenario | probabilities | entities (pooled) | mean predicted | observed purity | ECE | Brier | expected pure / seed | observed pure / seed | 95% interval covers | pairwise F1 @0.9 | pairwise F1 @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | old prior (t=0.9) | 1162 | 1.000 | 0.999 | 0.001 | 0.001 | 387.3 | 387.0 | 2/3 | 0.998 | 0.993 |
| default | old prior + TF (t=0.9) | 1162 | 1.000 | 0.999 | 0.001 | 0.001 | 387.3 | 387.0 | 2/3 | 0.998 | 0.995 |
| default | calibrated (t=0.5) | 1162 | 0.994 | 0.999 | 0.005 | 0.002 | 385.0 | 387.0 | 2/3 | 0.988 | 0.997 |
| default | calibrated + TF (t=0.5) | 1160 | 0.991 | 0.999 | 0.008 | 0.003 | 383.1 | 386.3 | 1/3 | 0.985 | 0.996 |
| default | calibrated + 30 labels (t=0.5) | 1162 | 0.995 | 0.999 | 0.004 | 0.002 | 385.6 | 387.0 | 2/3 | 0.988 | 0.997 |
| default | calibrated + joint fields (t=0.5) | 1162 | 0.994 | 0.999 | 0.005 | 0.002 | 385.0 | 387.0 | 2/3 | 0.988 | 0.997 |
| hard | old prior (t=0.9) | 1111 | 0.992 | 0.881 | 0.111 | 0.108 | 367.5 | 326.3 | 0/3 | 0.856 | 0.698 |
| hard | old prior + TF (t=0.9) | 1094 | 0.997 | 0.937 | 0.060 | 0.060 | 363.5 | 341.7 | 0/3 | 0.905 | 0.735 |
| hard | calibrated (t=0.5) | 1067 | 0.807 | 0.954 | 0.147 | 0.078 | 287.1 | 339.3 | 0/3 | 0.490 | 0.909 |
| hard | calibrated + TF (t=0.5) | 945 | 0.763 | 0.971 | 0.208 | 0.096 | 240.5 | 306.0 | 0/3 | 0.484 | 0.863 |
| hard | calibrated + 30 labels (t=0.5) | 1067 | 0.826 | 0.954 | 0.128 | 0.071 | 293.8 | 339.3 | 0/3 | 0.490 | 0.909 |
| hard | calibrated + joint fields (t=0.5) | 1067 | 0.807 | 0.954 | 0.147 | 0.078 | 287.1 | 339.3 | 0/3 | 0.490 | 0.909 |

![Pair-level reliability](uncertainty_calibration.png)