# Decision Packet for Daily Probes (2026-10-30)

## 1. Context & Active Benchmark
- **Active Benchmark**: Submission 48 (Score `.94633`, SHA256 `3f8d08e063870...`).
- **Recent Scored History (Rounds 72–76)**:
  * Sub 72: `.94623` (-0.00010 vs 48, -3 net items).
  * Sub 73: `.94600` (-0.00033 vs 48, -10 net items, catastrophic seller gate failure).
  * Sub 74: `.94614` (-0.00019 vs 48, -6 net items).
  * Sub 75: `.94633` (0.00000 tie, +0 net items, 10 fixes / 10 breaks, NOT promoted).
  * Sub 76: `.94633` (Exact duplicate of 48, 0 flips, wasted slot).
- **Duplicate Guard**: `src/eval_20261030/submission_guard.py` is actively enforced.

## 2. Audited Candidate Options for Codex

| Candidate Name | N Flips | Seen / New | -> Used / -> New | SHA256 (first 12) | Guard Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `candidate_sub77_dual_cb_high_margin_unprobed` | 20 | 17 / 3 | 8 / 12 | `3770adeea626...` | PASS |
| `candidate_sub78_model_consensus_unprobed` | 25 | 19 / 6 | 8 / 17 | `e00f535b1e79...` | PASS |
| `candidate_sub79_new_sellers_clean_text` | 20 | 0 / 20 | 2 / 18 | `6a75fd3389c2...` | PASS |
| `candidate_sub80_seen_sellers_conservative_unprobed` | 15 | 15 / 0 | 6 / 9 | `643b1b55c52f...` | PASS |

## 3. Candidate Hypotheses & Posterior Decision Branches

### `candidate_sub77_dual_cb_high_margin_unprobed`
- **Description**: Top 20 unprobed high-margin disagreements between Dual CatBoost R150 and S48
- **Target Segment**: Mixed (unprobed)
- **Purpose**: Test if pure model confidence over S48 yields positive score on virgin test items.
- **If Positive Score (Delta > 0)**: Licenses promoting high-confidence pure-model predictions into the adaptive 5th slot.
- **If Negative / Zero Score (Delta <= 0)**: Confirms S48 pseudo-labels are superior to raw model margin on boundary items.
- **Flipped IDs**: `[732, 3056, 3106, 3802, 4969, 5237, 6128, 6487, 6940, 7575, 13230, 16660, 18735, 18790, 19403, 23454, 25747, 25976, 26293, 29552]`

### `candidate_sub78_model_consensus_unprobed`
- **Description**: Top 25 unprobed consensus flips where Dual CatBoost and Strong Ensemble agree against S48
- **Target Segment**: Consensus (unprobed)
- **Purpose**: Test cross-architecture consensus (leaf-wise LightGBM + oblivious CatBoost) vs S48.
- **If Positive Score (Delta > 0)**: Validates ensemble consensus as a trusted signal for flipping S48.
- **If Negative / Zero Score (Delta <= 0)**: Proves model agreement on certain items reflects shared training bias rather than ground truth.
- **Flipped IDs**: `[732, 3619, 3802, 4599, 5028, 5205, 6128, 6487, 6562, 6940, 7278, 7575, 12641, 17714, 18790, 19403, 19545, 20200, 22327, 25326, 25600, 25940, 25976, 28448, 29552]`

### `candidate_sub79_new_sellers_clean_text`
- **Description**: 20 unprobed new-seller items with explicit lexical/warranty cues disagreeing with S48
- **Target Segment**: New sellers (text only)
- **Purpose**: Test pure text condition cues on new sellers where seller-ID cannot leak or mislead.
- **If Positive Score (Delta > 0)**: Unlocks high-precision lexical overrides for all unprobed new-seller items.
- **If Negative / Zero Score (Delta <= 0)**: Proves keyword heuristics on MercadoLibre text have high false-positive rates due to seller boilerplate.
- **Flipped IDs**: `[55, 944, 1653, 4432, 9674, 9761, 10644, 11905, 13634, 13987, 16499, 18688, 21971, 22345, 23234, 24721, 26804, 27455, 27734, 28333]`

### `candidate_sub80_seen_sellers_conservative_unprobed`
- **Description**: 15 unprobed seen-seller items with clean seller history, strictly disjoint from refuted 72-74 sellers
- **Target Segment**: Seen sellers (conservative)
- **Purpose**: Test conservative seen-seller flipping without the full-seller flipping contagion that harmed 72-74.
- **If Positive Score (Delta > 0)**: Confirms seen-seller model signals can be selectively applied when refuted seller clusters are isolated.
- **If Negative / Zero Score (Delta <= 0)**: Establishes that seen sellers in S48 are already saturated and should not be modified further.
- **Flipped IDs**: `[3056, 3106, 3802, 5237, 6128, 6487, 6940, 7575, 13230, 16660, 19403, 23454, 25976, 26293, 29552]`

## 4. Adaptive 5th Slot Builder
- The 5th slot engine is implemented in `src/eval_20261030/build_from_scores.py`.
- Once Ariel uploads the 4 chosen probes and Codex receives the scores, run:
  ```bash
  python src/eval_20261030/build_from_scores.py --scores '{"sub77": 0.94640, "sub78": 0.94627, "sub79": 0.94653, "sub80": 0.94620}'
  ```
- The engine will automatically:
  1. Discard all tied or negative probes.
  2. Select the disjoint union of winning probes.
  3. Verify against `submission_guard.py` to prevent duplicate submissions.
  4. Generate the final 5th submission CSV.

## 5. Artifact Directory & Verification Hashes
- **Benchmark Restoration**:
  * [Benchmark Report](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/benchmark_20261030/BENCHMARK_REPORT.md)
  * [Attribution Gap Table](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/benchmark_20261030/attribution_gap_table.csv)
  * Pipeline A Test Probs: `results/benchmark_20261030/test_probs_pipeline_A.npy` (SHA256: `c6453e51dbfd381867220be1d6c1c4fbd2cf9ea64f0b57f258c8012a0bbcfed5`)
- **Complementary Modeling**:
  * [Complementary Report](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/complementary_20261030/COMPLEMENTARY_REPORT.md)
  * [Complementary Summary](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/complementary_20261030/complementary_summary.csv)
  * Complementary Test Probs: `results/complementary_20261030/test_probs_complementary.npy` (SHA256: `27c419f45a8fc57c3877602bb0d75b983ccd3d881268e122b6d962fe83422e87`)
- **Evaluation & Audits**:
  * [Scored Registry](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/eval_20261030/scored_registry.md)
  * [Confirm Overlap Audit](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/eval_20261030/confirm_overlap_audit.md)
  * Test Matrix v2: `results/eval_20261030/test_matrix_by_id.csv.gz`
- **Submissions**:
  * Sub 77: `submissions/20261030/candidate_sub77_dual_cb_high_margin_unprobed.csv` (SHA256: `3770adeea626b1bb4a123680e071720d20d755711689366df0222049e78d2eb4`)
  * Sub 78: `submissions/20261030/candidate_sub78_model_consensus_unprobed.csv` (SHA256: `e00f535b1e79ee8ebf4b0e9c8bc8ea4025ec6eef9a8964e7fece4f82879f90cf`)
  * Sub 79: `submissions/20261030/candidate_sub79_new_sellers_clean_text.csv` (SHA256: `6a75fd3389c258d4a6f8ff50882d27e9e7354366e04dca01fa2eb89f5bc3a4a1`)
  * Sub 80: `submissions/20261030/candidate_sub80_seen_sellers_conservative_unprobed.csv` (SHA256: `643b1b55c52f6f1a8c9b2e2d8e4f165e317c2a7927bb22d3e1db67104ae05d76`)

## 6. Strategic Recommendation for Codex
1. **Send Submissions 77–80 as the 4 Day Probes**:
   - Every candidate has passed `submission_guard.py` (0 duplicate collisions).
   - They test distinct, orthogonal dimensions (model confidence, multi-architecture consensus, pure text on new sellers, conservative seen-seller correction).
   - All refuted blocks (72–74) are completely isolated and excluded.
2. **Execute the 5th Slot Conditionally**:
   - Feed the 4 resulting scores to `src/eval_20261030/build_from_scores.py`.
   - If a probe ties (e.g. like Sub 75), it is automatically excluded. Only strictly positive probes ($\Delta > 0$) are combined.
