# Separate field teachers

Adds title-only and warranty-only text experts to the deep tabular + combined NB teacher pipeline. Title uses character3–5 TF-IDF30k, warranty character2–5 TF-IDF15k, both log-count-ratio NB weighting and LogisticRegression C4. Each training row receives a strictly out-of-fold teacher score from3 stratified folds; vectorizer, ratio and classifier are fit only on that fold's training rows. Evaluation uses a teacher refit on the corresponding entire training stage. The combined teacher is reused from `training_round2/deep_teacher`, never generated redundantly here.

CatBoost depth7/750 max, lr.06/l2=15 with snapshots. Number of trees and threshold selected on7k tune before14k confirmation. Stage training is49k, then56k; only promising branches enter full70k refit under the budget. Field feature vocabulary is fixed before confirmation. Accuracy alone does not establish calibrated probabilities.

Inherited row partitions and earlier model choices were repeatedly used. Confirmation is not historically virgin, and this comparison is exploratory. The fixed historical85/15 blend is diagnostic, not a weight search. Historical reference is `refit49k_without_public` at threshold0.505.

Atomic NPZ and teacher joblib checkpoints, dataset+feature fingerprint, and CatBoost snapshots support resumption. Model inference on new rows requires the CatBoost model, full field teachers, parent combined teacher and identical deep/context feature construction; context stats are reconstructed from the original training data rather than separately serialized.
