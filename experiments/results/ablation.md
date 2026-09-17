# Schema-matching ablation

Default acceptance threshold 0.55; P/R/F1 micro-averaged per set; best F1 = mean over scenarios of the best threshold.

## fabricated-easy (8 scenarios)

| method | precision | recall | F1 @0.55 | best F1 | match time (s) |
|---|---|---|---|---|---|
| Exact name | 0.934 | 1.000 | 0.966 | 0.964 | 0.000 |
| A name | 0.934 | 1.000 | 0.966 | 0.964 | 0.002 |
| B +type | 0.467 | 1.000 | 0.637 | 0.964 | 0.001 |
| C +values | 0.885 | 1.000 | 0.939 | 0.972 | 0.041 |
| D +semantic | 0.885 | 1.000 | 0.939 | 0.977 | 0.089 |
| E1 +bipartite | 0.924 | 1.000 | 0.960 | 0.977 | 0.035 |
| E2 full graph | 0.885 | 1.000 | 0.939 | 0.970 | 0.049 |
| F learned | 1.000 | 1.000 | 1.000 | 1.000 | 0.086 |
| G blend | 0.977 | 1.000 | 0.988 | 1.000 | 0.073 |
| H MiniLM encoder | 0.885 | 1.000 | 0.939 | 0.970 | 0.089 |
| I bge-small encoder | 0.876 | 1.000 | 0.934 | 0.970 | 0.103 |

## fabricated-medium (8 scenarios)

| method | precision | recall | F1 @0.55 | best F1 | match time (s) |
|---|---|---|---|---|---|
| Exact name | 0.690 | 0.217 | 0.331 | 0.306 | 0.000 |
| A name | 0.674 | 0.630 | 0.652 | 0.726 | 0.003 |
| B +type | 0.361 | 0.804 | 0.498 | 0.750 | 0.002 |
| C +values | 0.829 | 1.000 | 0.906 | 0.929 | 0.038 |
| D +semantic | 0.844 | 1.000 | 0.915 | 0.926 | 0.041 |
| E1 +bipartite | 0.892 | 0.989 | 0.938 | 0.940 | 0.042 |
| E2 full graph | 0.902 | 1.000 | 0.948 | 0.947 | 0.051 |
| F learned | 1.000 | 1.000 | 1.000 | 1.000 | 0.065 |
| G blend | 0.989 | 1.000 | 0.995 | 1.000 | 0.075 |
| H MiniLM encoder | 0.902 | 1.000 | 0.948 | 0.947 | 0.086 |
| I bge-small encoder | 0.902 | 1.000 | 0.948 | 0.947 | 0.115 |

## fabricated-hard (8 scenarios)

| method | precision | recall | F1 @0.55 | best F1 | match time (s) |
|---|---|---|---|---|---|
| Exact name | 0.385 | 0.053 | 0.093 | 0.099 | 0.000 |
| A name | 0.195 | 0.232 | 0.212 | 0.309 | 0.003 |
| B +type | 0.182 | 0.505 | 0.267 | 0.337 | 0.002 |
| C +values | 0.772 | 1.000 | 0.872 | 0.894 | 0.030 |
| D +semantic | 0.767 | 0.968 | 0.856 | 0.900 | 0.039 |
| E1 +bipartite | 0.860 | 0.968 | 0.911 | 0.941 | 0.040 |
| E2 full graph | 0.861 | 0.979 | 0.916 | 0.936 | 0.045 |
| F learned | 0.989 | 0.989 | 0.989 | 0.989 | 0.066 |
| G blend | 0.969 | 0.989 | 0.979 | 0.989 | 0.076 |
| H MiniLM encoder | 0.836 | 0.968 | 0.898 | 0.916 | 0.085 |
| I bge-small encoder | 0.820 | 0.958 | 0.883 | 0.888 | 0.118 |

## sample (1 scenario)

| method | precision | recall | F1 @0.55 | best F1 | match time (s) |
|---|---|---|---|---|---|
| Exact name | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A name | 0.625 | 0.714 | 0.667 | 0.714 | 0.002 |
| B +type | 0.171 | 1.000 | 0.292 | 0.667 | 0.001 |
| C +values | 1.000 | 0.714 | 0.833 | 0.833 | 0.096 |
| D +semantic | 1.000 | 0.714 | 0.833 | 0.933 | 0.061 |
| E1 +bipartite | 1.000 | 0.714 | 0.833 | 0.933 | 0.060 |
| E2 full graph | 1.000 | 0.714 | 0.833 | 0.933 | 0.058 |
| F learned | 1.000 | 0.714 | 0.833 | 0.833 | 0.084 |
| G blend | 1.000 | 0.714 | 0.833 | 0.833 | 0.095 |
| H MiniLM encoder | 1.000 | 0.714 | 0.833 | 0.933 | 0.115 |
| I bge-small encoder | 1.000 | 0.714 | 0.833 | 0.933 | 0.141 |

## nyc (1 scenario)

| method | precision | recall | F1 @0.55 | best F1 | match time (s) |
|---|---|---|---|---|---|
| Exact name | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A name | 0.294 | 0.714 | 0.417 | 0.588 | 0.012 |
| B +type | 0.050 | 0.714 | 0.093 | 0.625 | 0.005 |
| C +values | 0.417 | 0.714 | 0.526 | 0.833 | 2.578 |
| D +semantic | 0.625 | 0.714 | 0.667 | 0.727 | 2.522 |
| E1 +bipartite | 0.833 | 0.714 | 0.769 | 0.769 | 2.545 |
| E2 full graph | 0.833 | 0.714 | 0.769 | 0.769 | 2.685 |
| F learned | 0.667 | 0.286 | 0.400 | 0.667 | 2.777 |
| G blend | 0.800 | 0.571 | 0.667 | 0.769 | 2.653 |
| H MiniLM encoder | 0.833 | 0.714 | 0.769 | 0.769 | 2.773 |
| I bge-small encoder | 0.833 | 0.714 | 0.769 | 0.769 | 3.009 |
