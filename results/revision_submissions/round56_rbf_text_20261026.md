# Round56 — RBF text specialist (2026-10-26)

A distinct second-stage model used 1,024 random Fourier RBF features over standardized 40-dimensional FastText embeddings trained on the original 100k texts, plus seller-group OOF word/char teacher predictions. Logistic regression was fit on the same 41,680 train rows as round39; tune 13,787 and confirm 14,533 have disjoint sellers from fit and each other. Baseline is the saved HistGB+FastText predictions at threshold .475; RBF model uses .5. Test labels and Kaggle scores were not used.

| Section | RBF accuracy | Baseline accuracy | Paired net RBF vs baseline | RBF / baseline log loss |
|---|---:|---:|---:|---:|
| Tune | .84377 | .92196 | −1,078 / 13,787 | .36097 / .20401 |
| Confirm | .83018 | .91523 | −1,236 / 14,533 | .38593 / .21063 |

Discard this RBF specialist and do not blend or submit it. It lacks structured information carried by the HistGB baseline; this is a limited test of one fixed kernel setting, not a rejection of every kernel model. Checkpoint `round56_rbf_checkpoint_20261026.zip`, Library ID `libfile_25b3223ffd108191bff93f49d3eca0bf`, SHA-256 `cebf718f520d5ab45936fb7743cbf5203c2228b66e99bd06220a737a321a3ea5`; includes code, paired probabilities, split IDs and JSON metrics. No Kaggle quota used.
