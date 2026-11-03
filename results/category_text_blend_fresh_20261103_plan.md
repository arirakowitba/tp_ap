# Fresh fixed-blend confirmation plan

The category-conditioned text OOF feature improved log loss but lost 12 and 3 accuracy decisions alone on development seeds 20261136/37. Exploratory inspection of their paired confirm probabilities found a small gain for a 25% probability blend with the historical model at threshold .500. Those two confirms are now **development**, not independent confirmation.

Before examining seed 20261138, freeze the candidate: `p_blend = .75*p_historical + .25*p_category_text`, decision `p_blend >= .500`. Comparator: `p_historical >= .500` on the same split. Train both models exactly as in `scripts/ablate_category_text_interaction_20261103.py` with fit/tune/confirm seller-group split seed 20261138; model early stopping may use tune, but blend weight and threshold do not. Evaluate once on confirm and report paired accuracy, log loss, fixes/breaks, seen/new, seller bootstrap and hashes. A small or uncertain result is not grounds for a public CSV.
