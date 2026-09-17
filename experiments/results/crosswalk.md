# Value crosswalks: different vocabularies for the same things

Column-match recall on the planted vocabulary correspondence (no static value map covers these vocabularies) and false links involving the vocabulary columns, with LLM-proposed crosswalks off/on. The negative control must stay unlinked.

| scenario | crosswalk | true correspondences | found | false links | crosswalks proposed | verified | dataset relationship |
|---|---|---|---|---|---|---|---|
| us_states | off | 1 | 0 | 0 | 0 | 0 | none |
| us_states | on | 1 | 1 | 0 | 1 | 1 | lookup 0.78 |
| countries | off | 1 | 0 | 0 | 0 | 0 | none |
| countries | on | 1 | 1 | 0 | 1 | 1 | lookup 0.74 |
| months | off | 1 | 0 | 0 | 0 | 0 | none |
| months | on | 1 | 1 | 0 | 1 | 1 | lookup 0.76 |
| negative_control | off | 0 | 0 | 0 | 0 | 0 | none |
| negative_control | on | 0 | 0 | 0 | 1 | 0 | none |