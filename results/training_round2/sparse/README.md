# Sparse text + structured LightGBM

Direct TF-IDF word unigrams/bigrams (30k) and character3–5grams(30k), no SVD. Sparse features include one-hot categoricals (minfrequency5), deterministic listing variables, OOF seller/seller×category target features, and train-only price context. Attributes mentioning CONDITION are excluded from text.

Uses inherited fit49k/tune7k/refit56k/confirm14k split. Vectorizers, category encoders, and seller/price maps fit only on the corresponding training rows. Two configurations selected by standalone tune accuracy; fixed85%native+15%sparse blend declared before confirmation. No public/test labels. Confirmation is historically reused, not virgin.

Script: `scripts/train_sparse_boost_round2.py`.
Log: `results/sparse_round2.log`.
State: training. Model/predictions saved after each stage; report.json is written only on completion. Full70k refit is gated on a fixed-blend confirmation improvement; no submission is automatically produced.
