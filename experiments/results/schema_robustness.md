# Relationship inference on unseen confusing schemas

80 generated scenarios (6 tables, meaningless names, 5 true relationships each). Correct = exactly the true relationships, the near-unique child reference inferred as N:1 in the right direction, and no accepted link between identifiers that share no values.

| identifier format | scenarios | fully correct | missed relationships | extra relationships | wrong N:1 direction | false identifier links | discovery (s) |
|---|---|---|---|---|---|---|---|
| uuid_hex | 20 | 20 | 0 | 0 | 0 | 0 | 0.15 |
| uuid_dashed | 20 | 20 | 0 | 0 | 0 | 0 | 0.13 |
| prefixed | 20 | 20 | 0 | 0 | 0 | 0 | 0.11 |
| integer | 20 | 20 | 0 | 0 | 0 | 0 | 0.12 |