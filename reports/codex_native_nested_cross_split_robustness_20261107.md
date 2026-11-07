# Cross-split robustness audit of native-text / nested-NB blend

Date UTC: 2026-11-07
Status: development-only; no Kaggle use; does not replace the ordered two-stage rerun.

## Inputs

Existing paired train-only artifacts:
- `replicate_native_mixed_20261106/{tune_pairs,pairs}.csv`
- `replicate_native_mixed_20261108/{tune_pairs,pairs}.csv`
- `replicate_native_group_20261107/{tune_pairs,pairs}.csv`

All comparisons use the same rows within each split and fixed threshold 0.50 unless stated otherwise. These artifacts predate the current provenance fix, so results are hypothesis support rather than acceptance evidence.

## Fixed-weight sweep on untouched confirm partitions

Net correct predictions against `cb_nested_nb`:

| native-text weight | mixed06 (n=12177) | mixed08 (n=12728) | pure07 (n=10510) | mean net rate | worst net rate |
|---:|---:|---:|---:|---:|---:|
| 0.40 | +28 | +8 | +29 | +0.001896 | +0.000629 |
| 0.45 | +34 | +15 | +27 | +0.002180 | +0.001179 |
| **0.50** | **+34** | **+17** | **+29** | **+0.002296** | **+0.001336** |
| 0.55 | +42 | +14 | +19 | +0.002119 | +0.001100 |
| 0.60 | +42 | +8 | +7 | +0.001581 | +0.000629 |
| 0.65 | +45 | +5 | +7 | +0.001585 | +0.000393 |

The frozen 50/50 blend is simultaneously best by average gain and by minimax gain among the tested weights. Aggregate development gain is +80 correct decisions across 35,415 confirm rows. This aggregation is descriptive because splits may overlap.

## Threshold transport failure

For the frozen 50/50 blend, thresholds selected independently on each tune partition transported as follows:

| split | tune-selected threshold | confirm accuracy selected | confirm accuracy at 0.50 | hit delta selected vs 0.50 |
|---|---:|---:|---:|---:|
| mixed06 | 0.57 | 0.948920 | 0.951302 | -29 |
| mixed08 | 0.40 | 0.947832 | 0.949167 | -17 |
| pure07 | 0.46 | 0.914082 | 0.912274 | +19 |

Threshold tuning loses 27 hits in aggregate across these confirm partitions and is directionally unstable. Keep threshold 0.50.

## Confidence override negative control

A pooled-tune confidence rule that overrides nested-NB with native-text only when native margin exceeds nested margin by 0.105 produced confirm gains +17, +4, +17. It improved all three splits but was uniformly weaker than the fixed 50/50 blend (+34, +17, +29). Do not add this router.

## Decision

- Preserve exact `0.50 * p_cb_native_text + 0.50 * p_cb_nested_nb`, threshold 0.50.
- Do not retune weight or threshold.
- Treat this audit as prior evidence only.
- Acceptance still requires the two ordered reruns executed after commit `0184622`, with code SHA embedded in results.
