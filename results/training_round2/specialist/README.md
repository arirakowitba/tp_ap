# Quantity specialist

A CatBoost expert trained only on initial_quantity<=1. Original fit49k/tune7k/confirm14k row split is reused. This is not a historically virgin holdout.

Two configurations are compared by subset tune accuracy (tie logloss). The chosen specialist is refitted on dev56k quantity<=1 before a single confirmation. Policies are declared in advance: specialist only when quantity<=1, native otherwise; and a fixed 50/50 specialist/native mixture for quantity<=1, native otherwise. Full refit is gated on either policy improving native confirmation accuracy; that gate does not establish statistical significance.

Input loading is streaming and retains selected raw fields only. No condition attributes or external test labels. Category/seller relative price statistics are fitted on the corresponding full fit/dev rows without labels. Seen-seller reporting is relative to native's complete fit/dev set, so it remains comparable.

Script: scripts/train_quantity_specialist_round2.py. Main native baseline probabilities are required from results/training_next/baseline_reproduction and results/new_models_20261021.
