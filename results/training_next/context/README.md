# Seller-context LightGBM

Run: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/train_seller_context_next.py`.

Two predeclared LightGBM configs, 1,600 maximum rounds and100-round early stopping, selected on historical 49kfit/7ktune. Refit selected iteration count on56kdev, inspect14kconfirm once. Confirmation is reused diagnostic data from prior work, not a new unseen estimate.

Seller and seller×category targets:5-fold cross-fitted fit encodings; held-out mappings from fit only. Features: counts, posterior means with prior strengths5/20, posterior entropy, reliability, interactions with initial_quantity1. Price context: mean/median/std/count fit only for category×currency and seller. Deterministic base/tabular features only; no text embedding or test labels.

Fixed diagnostic blend:85% CatBoost native+15% context. Gate for full70k refit: standalone or this fixed blend exceeds native confirmation accuracy. No public score selection. No corrections from probes.

Every tune config saves probabilities, booster and report; selected config/policy is locked before confirmation; confirmation predictions/model/report saved before optional full refit.
