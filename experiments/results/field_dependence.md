# Field dependence among matches (hard customer scenario, 3 seeds)

Lift = P(both fields exact) / (P(first exact) · P(second exact)). Lift ≈ 1 means the fields agree independently, so interaction terms cannot change the posterior.

| field pair | basis | pairs (weight) | P(first exact) | P(second exact) | P(both exact) | lift |
|---|---|---|---|---|---|---|
| name × city | true matches | 1379 | 0.645 | 0.936 | 0.601 | 0.996 |
| name × city | posterior-weighted | 1168 | 0.673 | 0.965 | 0.648 | 0.997 |
| name × email | true matches | 275 | 0.622 | 1.000 | 0.622 | 1.000 |
| name × email | posterior-weighted | 275 | 0.623 | 0.996 | 0.620 | 0.999 |
| name × phone | true matches | 246 | 0.695 | 1.000 | 0.695 | 1.000 |
| name × phone | posterior-weighted | 246 | 0.697 | 0.993 | 0.691 | 1.000 |
| city × email | true matches | 275 | 0.960 | 1.000 | 0.960 | 1.000 |
| city × email | posterior-weighted | 275 | 0.964 | 0.996 | 0.960 | 1.000 |
| city × phone | true matches | 246 | 0.955 | 1.000 | 0.955 | 1.000 |
| city × phone | posterior-weighted | 246 | 0.962 | 0.993 | 0.955 | 1.000 |
| email × phone | true matches | 67 | 1.000 | 1.000 | 1.000 | 1.000 |
| email × phone | posterior-weighted | 67 | 1.000 | 1.000 | 1.000 | 1.000 |

## Comparison patterns of pairs predicted 0.5–0.9 (name / city / email / phone)

| levels | pairs | mean predicted | observed match rate |
|---|---|---|---|
| exact / exact / null / null | 615 | 0.758 | 0.878 |
| high / exact / null / null | 244 | 0.642 | 0.898 |
| medium / exact / null / null | 71 | 0.511 | 0.592 |
| exact / different / null / exact | 5 | 0.857 | 1.000 |