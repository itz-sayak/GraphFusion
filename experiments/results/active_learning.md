# Active learning for entity resolution

Hard variant of the sample scenario (crm_email_missing=0.6, crm_phone_missing=0.6, mdm_email_missing=0.6, mdm_phone_missing=0.6); seeds [7, 11, 23]; oracle = ground truth; labels in batches of 10; record-link auto-merge threshold 0.5 (balanced mode, calibrated probabilities).

Variants: **constraints** = labels become must-/cannot-link constraints only; **+ m update** = labelled matches also add evidence to the Fellegi–Sunter m estimates; **+ calibration** = semi-supervised Platt recalibration of match weights using all labels; **+ random-only calibration** (app default) = recalibration uses only the randomly sampled labels. **hybrid** = each batch of 10 is 7 uncertainty-sampled + 3 random pairs.

| selection | label use | labels | precision | recall | pairwise F1 | labelled pairs that were matches | uncertain pairs left |
|---|---|---|---|---|---|---|---|
| uncertainty | constraints | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| uncertainty | constraints | 10 | 0.942 | 0.910 | 0.925 | 7.0 | 340 |
| uncertainty | constraints | 25 | 0.944 | 0.918 | 0.931 | 10.7 | 325 |
| uncertainty | constraints | 50 | 0.953 | 0.947 | 0.950 | 31.0 | 300 |
| uncertainty | constraints | 100 | 0.970 | 0.949 | 0.959 | 68.7 | 250 |
| random | constraints | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| random | constraints | 10 | 0.922 | 0.897 | 0.909 | 1.3 | 349 |
| random | constraints | 25 | 0.922 | 0.897 | 0.909 | 2.0 | 348 |
| random | constraints | 50 | 0.923 | 0.897 | 0.909 | 4.0 | 346 |
| random | constraints | 100 | 0.928 | 0.897 | 0.912 | 9.3 | 341 |
| uncertainty | constraints + m update | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| uncertainty | constraints + m update | 10 | 0.942 | 0.930 | 0.935 | 7.0 | 341 |
| uncertainty | constraints + m update | 25 | 0.948 | 0.935 | 0.941 | 9.3 | 326 |
| uncertainty | constraints + m update | 50 | 0.948 | 0.951 | 0.949 | 23.7 | 298 |
| uncertainty | constraints + m update | 100 | 0.967 | 0.952 | 0.959 | 65.0 | 252 |
| uncertainty | constraints + calibration | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| uncertainty | constraints + calibration | 10 | 0.942 | 0.930 | 0.935 | 7.0 | 344 |
| uncertainty | constraints + calibration | 25 | 0.954 | 0.888 | 0.920 | 11.0 | 325 |
| uncertainty | constraints + calibration | 50 | 0.961 | 0.914 | 0.937 | 28.7 | 392 |
| uncertainty | constraints + calibration | 100 | 0.963 | 0.951 | 0.957 | 67.3 | 622 |
| random | constraints + calibration | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| random | constraints + calibration | 10 | 0.922 | 0.897 | 0.909 | 1.3 | 430 |
| random | constraints + calibration | 25 | 0.922 | 0.897 | 0.909 | 2.0 | 429 |
| random | constraints + calibration | 50 | 0.924 | 0.916 | 0.920 | 4.0 | 348 |
| random | constraints + calibration | 100 | 0.928 | 0.897 | 0.912 | 9.3 | 420 |
| hybrid | constraints + random-only calibration | 0 | 0.922 | 0.897 | 0.909 | 0.0 | 350 |
| hybrid | constraints + random-only calibration | 10 | 0.935 | 0.906 | 0.920 | 5.3 | 342 |
| hybrid | constraints + random-only calibration | 25 | 0.944 | 0.913 | 0.927 | 8.3 | 332 |
| hybrid | constraints + random-only calibration | 50 | 0.945 | 0.929 | 0.937 | 19.0 | 315 |
| hybrid | constraints + random-only calibration | 100 | 0.960 | 0.948 | 0.954 | 47.0 | 280 |

![F1 vs labels](active_learning.png)