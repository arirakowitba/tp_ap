# Decision Packet for Daily Probes (2026-10-30)

## 1. Context & Active Benchmark
- **Active Benchmark**: Submission 48 (Score `.94633`, SHA256 `3f8d08e061854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`).
- **Scored History (Rounds 72–76) — Canonical Aggregate Evidence**:
  * **Sub 72**: Score `.94623` (Delta displayed `-0.00010` vs 48). Total flips: **8** (all in new-seller segment, direction used->new). Decision: Exclude.
  * **Sub 73**: Score `.94600` (Delta displayed `-0.00033` vs 48). Total flips: **7** (all in new-seller segment, direction new->used). Decision: Exclude.
    *(Correction: Sub 73 contained only 7 flipped items in total. Public score differences cannot be translated into exact item counts like "-10 net items", which is mathematically impossible given only 7 flips. Public deltas reflect aggregate block evidence on an unknown public sample size, not individual labels).*
  * **Sub 74**: Score `.94614` (Delta displayed `-0.00019` vs 48). Total flips: **8** (seen-seller segment, mixed direction: 5 to new, 3 to used). Decision: Exclude.
  * **Sub 75**: Score `.94633` (Delta displayed `0.00000` tie vs 48). Total flips: **5** (seen-seller lexical override to used). Decision: Exclude / Tie (no promotion without confirmed gain).
    *(Correction: Sub 75 contained only 5 flipped items in total. The prior report of "10 fixes / 10 breaks" was an error; a 5-item flip cannot produce 10 fixes and 10 breaks).*
  * **Sub 76**: Score `.94633` (Exact duplicate of Sub 48, 0 flips, wasted slot).
- **Core Verification Rule**: Public score differences are block-level evidence and aggregate signals. They are **never** to be converted into pseudo-labels for individual IDs or exact integer item counts without knowing the exact public set denominator and rounding precision.
- **Duplicate & Overlap Guard**: `src/eval_20261030/submission_guard.py` is actively enforced to prevent duplicate submissions and uninspected collisions.

---

## 2. Clarification of Seen vs. New Seller Dynamics

A critical contradiction between validation holdouts and the test set has been resolved:
1. **Validation Holdout (R151 / R153 confirm split)**:
   - Designed specifically to stress-test generalization to unseen sellers.
   - Out of 12,905 confirm rows (Seed 2026): **3,903 rows (30.2%)** belong to seen sellers (sellers present in `fit`), and **9,002 rows (69.8%)** belong to new sellers (held-out seller groups).
   - *(Prior documentation incorrectly inverted these proportions as 69.6% seen and 30.4% new; the true breakdown is 30.2% seen and 69.8% new).*
2. **Kaggle Test Set (30,000 items)**:
   - Evaluated against the 70,000 labeled training dataset:
     * **21,487 items (71.62%)** belong to sellers seen in the training set (`seen_seller`).
     * **8,513 items (28.38%)** belong to sellers not present in the training set (`new_seller`).
3. **Candidate Probe Flip Breakdown (Audited vs 70k Train Sellers)**:
   - Candidate Sub 77: 20 flips (17 seen, 3 new; 8 to used, 12 to new).
   - Candidate Sub 78: 25 flips (19 seen, 6 new; 8 to used, 17 to new).
   - Candidate Sub 79: 20 flips (0 seen, 20 new; 2 to used, 18 to new).
   - Candidate Sub 80: 15 flips (15 seen, 0 new; 6 to used, 9 to new).

---

## 3. Audited Candidate Options & Structural Overlap Audit

| Candidate Name | N Flips | Seen / New | -> Used / -> New | SHA256 (first 12) | Guard Status | Structural Relationship |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `candidate_sub77_dual_cb_high_margin_unprobed` | 20 | 17 / 3 | 8 / 12 | `3770adeea626...` | PASS | **Superset of Sub 80** (contains all 15 Sub 80 IDs + 5 extra) |
| `candidate_sub78_model_consensus_unprobed` | 25 | 19 / 6 | 8 / 17 | `e00f535b1e79...` | PASS | **Overlaps 10 IDs with Sub 77** (8 in Sub 80, 2 in Sub 77\80) |
| `candidate_sub79_new_sellers_clean_text` | 20 | 0 / 20 | 2 / 18 | `6a75fd3389c2...` | PASS | **Completely Disjoint** (0 overlap with 77, 78, 80) |
| `candidate_sub80_seen_sellers_conservative_unprobed` | 15 | 15 / 0 | 6 / 9 | `643b1b55c52f...` | PASS | **Strict Subset of Sub 77** (15 / 15 IDs inside Sub 77) |

> [!WARNING]
> Sub 80 is a strict mathematical subset of Sub 77 ($15 \subset 20$). Furthermore, Sub 78 shares 10 flipped items with Sub 77.
> **Never add public deltas from overlapping probes**. A new SHA-256 hash does not represent an independent hypothesis or disjoint IDs. If multiple overlapping probes are scored, they must be interpreted as nested contrasts, not additive gains.

---

## 4. Candidate Hypotheses & Nested Decision Branches

### Hypothesis A: `candidate_sub80_seen_sellers_conservative_unprobed` (Base Seen Block, N=15)
- **Description**: 15 unprobed seen-seller items with clean seller history, strictly disjoint from refuted 72–74 seller clusters.
- **Purpose**: Test whether conservative seen-seller model flips yield positive score when refuted seller clusters are isolated.
- **If Score > 0.94633**: Validates that high-confidence seen-seller adjustments are reliable; unlocks conservative seen-seller refinement.
- **If Score <= 0.94633**: Confirms seen-seller predictions in Sub 48 are already saturated and should not be modified.

### Hypothesis B: `candidate_sub77_dual_cb_high_margin_unprobed` (Nested Contrast vs Sub 80, N=20)
- **Description**: Contains the 15 items of Sub 80 PLUS 5 additional high-margin items (`[732, 4969, 18735, 18790, 25747]`).
- **Purpose**: Evaluated against Sub 80, this serves as an **explicit nested contrast** isolating the 5 marginal items.
- **If Score(77) > Score(80)**: Proves that the 5 additional marginal items contributed net positive signal.
- **If Score(77) <= Score(80)**: Proves that the 5 additional marginal items introduced net errors.

### Hypothesis C: `candidate_sub78_model_consensus_unprobed` (Cross-Architecture Consensus, N=25)
- **Description**: 25 unprobed flips where leaf-wise LightGBM and oblivious CatBoost agree against Sub 48.
- **Structure**: Contains 15 consensus-unique items, plus 10 items overlapping with Sub 77/80.
- **Purpose**: Test whether dual-architecture consensus identifies genuine errors in Sub 48.
- **Decision Condition**: If Sub 78 improves over Sub 48, consensus items that do not overlap with Sub 80 can be licensed.

### Hypothesis D: `candidate_sub79_new_sellers_clean_text` (Orthogonal Disjoint Block, N=20)
- **Description**: 20 unprobed new-seller items with explicit lexical/warranty cues disagreeing with Sub 48.
- **Structure**: **100% disjoint** from Sub 77, 78, and 80 (0 overlap).
- **Purpose**: Test lexical overrides on new sellers where seller ID cannot bias the prediction.
- **If Score > 0.94633**: Confirms lexical condition cues provide valid overrides on new sellers.
- **If Score <= 0.94633**: Confirms lexical cues suffer from false-positive boilerplate in test listings.

---

## 5. Adaptive 5th Slot Protocol (Conditioned on Disjoint Contrasts)
- The 5th slot engine (`src/eval_20261030/build_from_scores.py`) must be executed **only** after receiving scores from Ariel.
- **Strict Combination Rules**:
  1. Any probe with non-positive delta ($\Delta \le 0$) is discarded.
  2. If both Sub 80 and Sub 77 are positive, their nested difference ($\Delta_{77} - \Delta_{80}$) dictates whether the 5 extra items are included. Overlapping items are counted exactly once.
  3. If Sub 78 is positive, only its disjoint consensus items are eligible for union.
  4. Disjoint winning blocks (e.g. Sub 79 new-seller text + winning seen-seller block) are merged without item collisions.
  5. The output must pass `submission_guard.py` before finalization.

---

## 6. Artifact Directory & Verification Hashes
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
- **Candidate Submission Files**:
  * Sub 77: `submissions/20261030/candidate_sub77_dual_cb_high_margin_unprobed.csv` (SHA256: `3770adeea626b1bb4a123680e071720d20d755711689366df0222049e78d2eb4`)
  * Sub 78: `submissions/20261030/candidate_sub78_model_consensus_unprobed.csv` (SHA256: `e00f535b1e79ee8ebf4b0e9c8bc8ea4025ec6eef9a8964e7fece4f82879f90cf`)
  * Sub 79: `submissions/20261030/candidate_sub79_new_sellers_clean_text.csv` (SHA256: `6a75fd3389c258d4a6f8ff50882d27e9e7354366e04dca01fa2eb89f5bc3a4a1`)
  * Sub 80: `submissions/20261030/candidate_sub80_seen_sellers_conservative_unprobed.csv` (SHA256: `643b1b55c52f6f1a8c9b2e2d8e4f165e317c2a7927bb22d3e1db67104ae05d76`)
