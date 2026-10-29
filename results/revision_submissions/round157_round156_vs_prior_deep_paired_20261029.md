# Round157 — round156 ensemble versus prior deep CatBoost

2026-10-29. Main reviewed through fbb4be7. Joined paired_ensemble_seed_{seed}.csv with paired_deep_ablation_seed_{seed}.csv by id. All 12,905 / 12,843 confirm IDs match, all labels and seen masks match. Compare pred_ensemble with prior pred_base (thresholds already frozen in the producing experiments); log loss uses p_ensemble and prior p_base.

| Seed | Fixes | Breaks | Net versus prior deep | Prior deep LL | Ensemble LL |
|---:|---:|---:|---:|---:|---:|
|2026|221|410|**-189**|.187637|.221770|
|314159|222|416|**-194**|.195584|.228825|

The +56/+88 reported by round156 is relative to its own weaker CatBoost. Against the existing deep baseline on exactly the same labeled IDs, the ensemble loses substantially in both accuracy and log loss. Do not promote it or refit it for Kaggle as a stronger base until the baseline regression is explained. Identical confirm rows do not prove identical fit procedures; audit fit IDs, feature extraction, text teachers, target encodings and hyperparameters. No public scores were interpreted as per-ID labels. These historical confirms overlap and are not independent evidence.
