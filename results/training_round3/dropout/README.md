# Seller dropout on deep features and OOF text teacher

The experiment tests whether reducing seller identity dependence improves transfer to previously unseen sellers.

- Existing fit49k/tune7k/confirm14k row partitions are reused.
- Teacher probabilities are reused from the deep-teacher pipeline. Training probabilities remain strictly three-fold OOF; no seller information is used by the NB text teacher.
- Two train-only dropout rates, 20% and 50%, are compared. A fixed random draw keyed by global row index makes masks nested and reproducible across fits.
- Masked seller IDs become unique `__drop_row_ID` categories. Seller price mean/std/count become -1 and relative price becomes log_price+1, matching the pipeline's fallback for unseen sellers. Geography remains observed.
- Tune and confirmation retain real seller identity and their proper training-only seller statistics.
- Tree count uses tune early stopping; threshold and dropout fraction use tune accuracy weighted 71.6% seen and 28.4% new, with logloss as a tie-breaker.
- Configuration is locked before confirmation. No public labels, recovered labels, or flip feedback are used.
- Snapshots every30seconds are kept separately by stage/config. Re-running the unchanged data/params resumes training; completed tune predictions are cached.
- Full refit reuses the parent full teacher cache only if already available, and only if the model improves the relevant deep/native comparison within the time budget.

These inherited row partitions and earlier model designs have been repeatedly used. This is an exploratory ablation, not historically virgin confirmation. A change in weighted accuracy may trade seen-seller accuracy for unseen-seller performance.

A post-launch code change added retry handling for partially written teacher NPZ files; it does not alter learning parameters or inputs. The initial executed script hash remains recorded in source_integrity.json.
