# Review after overnight recovery: subset promotion is not identified

Reviewed main f68146686aedf964ee275e2adb3115a307bc93c2 and Claude 72c790c on 2026-10-30. No later changes found at review time.

## Concrete counterexample for build_from_scores.py

The legacy handling of positive candidate78 extracts its 15 unique flips when candidate77/80 also wins. Positive aggregate evidence for the full block does not identify that subset's contribution.

Synthetic example in arbitrary additive accuracy units (NOT actual Kaggle labels or inferred counts): let overlap O contribute +10, candidate78-only subset U contribute -5, and the rest of candidate77 R contribute -4. Then score changes are O+U=+5 for candidate78 and O+R=+6 for candidate77. Both probes win. The implemented union O+R+U contributes only +1, worse than either winner. Removing double-counting does not fix this identification problem. The observed full-block scores alone cannot establish the sign of U.

Required behavior: stop or require a separately scored disjoint block U / identifiable nested contrast before treating U as a publicly validated improvement. Never infer per-ID labels.

## Redesign compatibility

New candidate77_nested_margin5 contains only the 5 marginal IDs; old candidate77 contains 20 IDs including the 15-ID candidate80 core. Current filename matching for '77' applies the legacy superset logic to both. This can reject a beneficial independent marginal block or misdescribe its contrast. Identify contracts through exact hashes and ID/target maps, never filename substrings. Verify subset relations dynamically.

## Model review priorities

The 0.75 CatBoost / 0.25 LightGBM results (+26,+56,+4; seller bootstrap intervals [-2,56],[27,85],[-17,25]) remain development evidence on reused confirms. Freeze the policy and evaluate honest outer splits. Separate threshold changes from probability-model changes. Lower log loss is not by itself a calibration proof.

The baseline report's exact-reproduction assertion conflicts with the new manifest's agreement near 98.3%. Version historical and current artifacts separately and resolve the discrepancy before promotion.

Neural pilot comparison uses baseline stored predictions and blend threshold .50; isolate threshold effects. Its GroupKFold OOF reuses vocabulary/IDF fitted on all fit rows, so describe that transductive preprocessing accurately or fit vectorizers within folds for strict inductive OOF.

No new model was trained in this review. No submission is promoted. Submission48 remains the demonstrated public fallback. This report is a reproducible logical audit, not a claim of new predictive gain.
