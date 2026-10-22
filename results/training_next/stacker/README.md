# Tune-only regularized probability stacker

Script: `scripts/train_regularized_stacker_next.py`.

Five base predictors: native CatBoost, NB-weighted sparse text, global TF-IDF retrieval k=128 similarity power2, rich feature CatBoost config1, and contextual LightGBM config1. Inputs are their clipped logits plus seen-seller, quantity <=1, free listing, log quantity, and three fixed gate interactions with each base's logit difference from native.

The native tune vector is a dependency; the script waits up to ten minutes with status checkpoints. It never substitutes native full-fit predictions on training rows. Every input's row order is checked against fixed tune/confirm partitions.

Meta logistic regression is standardized inside the five-fold inner tune CV. C is selected among .01/.1/1 by mean accuracy, breaking ties toward smaller C. The chosen meta model is fit on tune7k and locked before confirmation is loaded. Base predictors fit49k produce tune inputs, then base refits56k produce confirm14k inputs. This base-refit distribution shift is a limitation, and full70k test bases have another shift.

The inner CV score is selection machinery, not an unbiased validation estimate: individual base hyperparameters already used tune labels. Confirmation is inherited and has been reused for prior experiments. No public-original labels or inferred flip labels are permitted.

Test probabilities are exported only when all five full70k base vectors exist. Otherwise the model and exact pending dependency are checkpointed.
