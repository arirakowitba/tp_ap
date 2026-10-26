# Round63 — native CatBoost text architecture (2026-10-26)

New hypothesis: learn directly from original raw text using CatBoost native text processing, combined with structured numeric/categorical inputs and the original100k unsupervised40d FastText embeddings. Seller identity excluded. This replaces OOF text teacher inputs with native text processing; it is a distinct architecture and its loss cannot be attributed solely to one text calculator. CatBoost600iterations/depth6/lr.045/l2=12/seed6301/2threads;129 input columns. Fit41680/tune13787/confirm14533 seller-disjoint within split, historically reused split. Candidate threshold .5; references HistGB+FT .475, CatBoost OOF .5. No test labels.

| Section | Accuracy | Log loss | Net vs HistGB+FT (seller95 interval) | Net vs Cat OOF (seller95 interval) |
|---|---:|---:|---:|---:|
| Tune |.915790|.212181|−85 [−135,−35]|−43 [−97,+6]|
| Confirm |.909792|.221457|−79 [−134,−23]|−91 [−150,−33]|

Discard as replacement base; no submission or post-hoc consensus rule. It fixes240 baseline errors but breaks319 in confirm, so raw complementarity alone does not justify flips. Bootstrap2000 seller resamples seed6301. Further text architectures require a new hypothesis; avoid reusing this confirm to tune the same model.

Checkpoint `round63_native_text_checkpoint_20261026.zip`, Library ID `libfile_35a5f797bd3881918d82e4c84c2740f3`, SHA256 `16abd84ba17aa0f9af92d305fbe471a93ea9fb4482a8744a05f725cb96ba794b`. Includes trained .cbm, code, paired probabilities/split IDs, metrics and hashes. No Kaggle quota. Public best remains41=.94628.
