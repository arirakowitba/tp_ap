# Round61 — validation coverage audit (2026-10-26)

Four recent seller-disjoint split realizations cover 61,181 of70,000 train rows and24,328 sellers in the union of tune+confirm holdouts. Cumulative row coverage is28,320,45,971,55,206,61,181. There are8,819 rows outside this union.

These8,819 rows are only outside the FOUR recent holdouts. They are not verified pristine because earlier project validation partitions have not yet been cross-referenced. No labels of this pool were inspected to choose a new rule. Do not treat another random cut of all70k as independent confirmation of models already selected on overlapping cuts. A seller split is disjoint within one fit/tune/confirm, but that does not make multiple split realizations independent.

Next gate: cross-reference all older preserved partition IDs before assigning a final confirmation pool, then freeze a candidate and baselines before opening that pool. If no untouched pool survives, explicitly label subsequent results as reused validation and reserve stronger claims for external evaluation by Ariel.

Checkpoint `round61_validation_coverage_20261026.zip`, Library ID `libfile_4ec1d47282c481918cedb18ecdc85c70`, SHA256 `6dd5df2ea5fe05eb1b0b0b5d5d21dbb2e212d5f7f7c19b393eb5524cad721e96`. Contains coverage and scoped outside-union row IDs. This was a validation audit; no new training or Kaggle submission.
