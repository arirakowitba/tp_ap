# Round62 — historical validation coverage extension (2026-10-26)

Extended round61's four recent cuts with preserved holdout IDs in round11_12 and round13 checkpoints. The8,819 rows outside the four recent holdouts shrink to6,189 after round11 tune/confirm, and4,965 after round13 valid. Duplicate round12/model holdouts add no rows. Keys were checked explicitly: round13 uses `valid`, not `confirm`.

Excluding every seller having ANY row in the audited holdout union leaves **1,897 rows from1,647 sellers**. This exclusion used seller IDs only; labels were not used to select the pool. This is still NOT a verified pristine pool: older partitions beyond the audited archives are not exhaustively cross-referenced. A pool this small also has limited power for gains of a few cases; do not spend it repeatedly on model selection.

Checkpoint `round62_historical_validation_coverage_20261026.zip`, Library ID `libfile_074cb1eb79f88191b981093faaacdfbb`, SHA256 `ad177e5d1e9b547751ab6a0c9a8a36931f26082e22df3e8bc74b4f4e3ccafb20`. Includes source-by-source coverage, outside-union IDs and seller-exclusive IDs with scoped limitations. No new model training, test labels or Kaggle quota.
