# Joint representation experiment

This experiment appends three strictly out-of-fold probabilities to the existing structured/text representation: LogisticRegression over TF-IDF, LinearSVC over TF-IDF, and LightGBM over structured features with fold-local encoding. Training probabilities come from `stacking_seed42_base.npz` inner grouped folds; holdout probabilities come from models fitted on the corresponding complete outer training partition. Learned meta predictions evaluated on the same training rows are never inserted as features.

Ablations keep stacking features and optionally replace the old SVM probability with the normalized-text OOF score, add eight domain flags, and append 384 frozen pretrained multilingual title embedding coordinates. Embeddings are unsupervised pretrained vectors, normalized to unit length; they contain no competition target training.

Every candidate JSON records its optional blocks, fitted number of boosting iterations, threshold and parameters. Prediction CSV files contain exact global row IDs and labels. The loader asserts that the split and fingerprint agree with the baseline cache and strict stacking cache. `prepare_joint` is reusable by the confirmation/refit pipeline with newly constructed fold-local inputs.

TPE explores six LightGBM parameters on the frozen seed42 grouped selection partition. Its accuracy is selection accuracy, not a Kaggle score. Only a candidate frozen before the separate seed731 check can qualify for promotion; the separate split is a robustness check on overlapping data, not an independent dataset.
