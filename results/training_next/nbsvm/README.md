# NB-weighted sparse text challenger

Reproduction: `python -u scripts/train_nbsvm_next.py` from repository root.

The representation concatenates word TF-IDF (1–2 grams, 100k features) and character TF-IDF (3–5 grams, 100k features). Corpus includes title (double weight), warranty, attribute names/values and category token. No seller IDs, item IDs, URLs, public-original labels or inferred flip labels enter the model.

The log-count ratio is log[(sum positive TF-IDF+1)/positive mass divided by (sum negative TF-IDF+1)/negative mass], fitted on training rows only. Logistic regression then fits reweighted sparse TF-IDF. This tests sentiment-style NB weighting without assuming literal condition words dominate.

C is selected from .25, 1, 4 using accuracy on the existing 7k tune set, breaking ties by log loss. Refit on 56k, confirm on the existing 14k set. Report seen/unseen sellers relative to the 56k development rows. These are inherited row splits and are not a fresh external estimate after repeated experimentation.

If baseline tune probabilities exist, choose native/NB weights on tune. Otherwise an 85% native / 15% NB blend is declared before confirmation and evaluated without searching confirmation weights. Full refit is gated on this blend improving confirmation accuracy; this gate does not itself establish public leaderboard improvement.

Every completed C trial writes a progress report and tune probability NPZ. The final report records the exact selection path, confirmation vectors, and whether full refit ran.

## Historical matched comparison

`evaluate_nb_historical_matched.py` compares the predeclared 85% historical / 15% NB blend over 4,147 rows excluded from both models' respective training sets. Against the historical matched-without-public composition, accuracy changes 93.9233% → 93.9957% (+3; 14 gains and 11 losses, unadjusted paired p=.690). Against the separate historical without-public composition, both are 93.9957% (9 gains / 9 losses). Log loss worsens in both comparisons.

Thus the NB challenger shows complementary errors against the newly trained weaker native baseline, but does not supply convincing evidence of improvement over the stronger historical composition. Do not prioritize a public submission solely on the +27/14k result. Retain NB probabilities as a candidate input for independently selected ensembles and targeted diagnostics.
