# Round57 — RBF with matched structured inputs (2026-10-26)

This corrects the information mismatch in round56. The candidate uses the same 145 tabular, FastText and seller-group OOF word/char predictor inputs as HistGB+FastText. StandardScaler fit on fit only, clipping [-6,6], 1024 RBF random features gamma .004 (seed5701), logistic C10. Same seller-disjoint fit41680/tune13787/confirm14533; parameters fixed before opening confirm. Existing split has prior reuse, so it is not pristine independent evidence.

| Section | HistGB+FT accuracy | RBF accuracy | Paired net | Log loss baseline / RBF |
|---|---:|---:|---:|---:|
| Tune | .921955 | .902880 | −263 | .204014 / .240592 |
| Confirm | .915227 | .890869 | −354 | .210628 / .267548 |

Baseline threshold .475, candidate .5. Close this fixed RBF architecture; do not submit or blend. Adding structured information helped relative to text-only round56 but did not reach the reference. This does not reject all kernels or establish effects on unseen test labels.

Checkpoint `round57_allinput_rbf_checkpoint_20261026.zip`, Library ID `libfile_ba2f905001588191abd470f0ee74119e`, SHA256 `6d2307c11c510b0efa5f442759d4c7127660f5f562986b503c253de005d52872`. Contains complete paired probabilities, split IDs, code, report, hashes. No Kaggle quota used. Public best remains 41=.94628.
