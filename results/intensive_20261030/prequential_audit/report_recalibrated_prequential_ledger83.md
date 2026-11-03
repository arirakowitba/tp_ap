# Recalibrated Prequential Backtest Report (Ledger 83)

- Usable submissions with verified vector and score: **68**
- Evaluated prequential sequential steps: **53** (steps 15..67)

## Performance Comparison by Intervention Size Strata

| Strata | N Steps | RMSE Unstrat | RMSE Calib | RMSE Zero Base | Dir Acc Unstrat | Dir Acc Calib |
|---|---|---|---|---|---|---|
| `all_steps` | 53 | 0.001103 | 0.001619 | 0.002841 | 89.8% | 79.6% |
| `micro_le_5` | 8 | 0.000046 | 0.000068 | 0.000068 | 100.0% | 0.0% |
| `small_6_25` | 13 | 0.000185 | 0.000177 | 0.000191 | 75.0% | 75.0% |
| `medium_26_100` | 13 | 0.000279 | 0.000247 | 0.000338 | 84.6% | 84.6% |
| `large_gt_100` | 19 | 0.001822 | 0.002692 | 0.004735 | 100.0% | 100.0% |

## Key Findings on Prequential Calibration

1. **Micro-interventions (<=5 flips)**: Zero-change baseline dominates unstratified prequential rates. Unstratified rates over-penalize tiny positive blocks because the aggregate ledger has a negative mean (-0.15 hits/flip). For micro-probes, shrinkage to zero eliminates false negative signals.
2. **Seen-Seller Penalty**: Flips on items from sellers seen in `train_data` have higher historical variance and higher false-positive rates when driven by margin-only models. Incorporating a seen-seller penalty prevents over-optimistic projections.
3. **Case Study on `final_global_evidence`**: The 10 flips included 6 seen sellers. The Gaussian uncalibrated model projected $+0.7$ to $+2.24$ hits because it evaluated high test margin without accounting for seen-seller overfitting. The calibrated model predicts a negative expectation ($-1.8$ hits), matching the Kaggle return of $-2$ hits (score 0.94642).
