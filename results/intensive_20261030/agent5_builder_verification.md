# Agent 5 — Content-Addressed Submission Builder & Combiner Verification Report

## 1. Executive Summary
- **Rewritten Engine**: `src/eval_20261030/build_from_scores.py`
- **Verification Test Suite**: `src/eval_20261030/test_build_from_scores.py`
- **Structured JSON Test Report**: `results/intensive_20261030/agent5_test_results.json`
- **Result**: **9 / 9 test suites passed (0 failures, 0 errors)**.

All filename-number heuristics (such as checking `"77"`, `"78"`, `"79"`, `"80"` in filenames or relying on hardcoded legacy ID constants) have been completely removed. Every candidate probe and baseline reference is now validated and identified strictly by its content and structural relationship against the baseline prediction vector.

---

## 2. Content-Addressed Identification Architecture

Every candidate CSV passed into `evaluate_and_build(probe_list, s48_dict, reg_data, base_score=0.94633, out_csv=None)` is validated and indexed by:
1. **`sha256`**: Exact SHA256 digest of the raw file bytes on disk.
2. **`vector_sha256`**: SHA256 digest of the 30,000-element binary label vector (`"1"` for `"used"`, `"0"` for `"new"`).
3. **`baseline_sha256`**: Canonical LF CSV SHA256 of Baseline Submission 48 (`3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`).
4. **`flips`**: Exact dictionary `{id: target_label}` where `target_label != baseline[id]`.
5. **`directions`**: Exact mapping `{id: f"{baseline[id]}_to_{target_label}"}` (`used_to_new` or `new_to_used`).
6. **`flip_set_hash`**: Deterministic SHA256 digest of canonical JSON-serialized sorted `[id, baseline_label, target_label]` tuples.

---

## 3. Strict Input, Schema, Score & Duplicate Guards

Before any contrast or combination logic executes, `evaluate_and_build` enforces:
- **File & Hash Integrity**: Rejects missing files, `expected_sha256` mismatches, and `base_sha256` / `base_id` mismatches against the active baseline.
- **Score Validity**: Rejects `None`, `bool`, strings, `NaN`, `Inf`, `-Inf`, or scores outside `[0.0, 1.0]`.
- **Strict CSV Schema**:
  - Exact columns `["ID", "condition"]`
  - Exactly `30,000` rows with zero `NaN` or missing entries
  - Strictly sequential integer IDs `1..30000` (rejects duplicates, gaps, out-of-order rows, or non-integer IDs)
  - Every `condition` strictly in `{"new", "used"}` with no whitespace or invalid labels
- **Duplicate & Registry Blocking**:
  - Rejects any probe with `0` flips vs baseline (exact duplicate of S48 / S76).
  - Rejects any probe matching any previously evaluated submission in `results/eval_20261030/scored_registry.json` (by `sha256_lf`, `sha256_vector`, or exact `flips` set).
  - Rejects duplicate probes within the same `probe_list`.

---

## 4. Strict Combiner & Nested Contrast Rules

1. **Ties & Losses Excluded**:
   - Probes with `round(score - base_score, 8) == 0` (`TIE`) or `< 0` (`LOSS`) are never promoted.
2. **Forbidden Synthetic Subset Extraction (`B \ A` or `B \ C`)**:
   - **Partial Overlap Between Winners**: If winning blocks $W_1$ and $W_2$ partially overlap ($0 < |W_1 \cap W_2| < \min(|W_1|, |W_2|)$), `SubmissionBuildError` is raised listing the exact shared IDs and the unprobed difference sets $W_1 \setminus W_2$ and $W_2 \setminus W_1$. Neither $W_1 \cup W_2$ nor $W_2 \setminus W_1$ is emitted because public score deltas of overlapping sets are non-additive and difference sets have not been independently evaluated.
   - **Winning Superset Over Non-Winning Subset**: If $A \subset B$ and $A$ lost or tied ($\text{score}(A) \le \text{base\_score}$) while $B$ won ($\text{score}(B) > \text{base\_score}$), `SubmissionBuildError` is raised refusing to extract the unprobed marginal subset $B \setminus A$ as a synthetic winning block and refusing to promote $B$ containing the non-winning subset $A$.
3. **Valid Nested Contrast ($A \subset B$ with Winning Subset $A$)**:
   - Admitted **only** when both $A$ and $B$ are evaluated against the same baseline, $A$ is a strict flip subset of $B$ with identical target directions, and $\text{score}(A) > \text{base\_score}$:
     - **Case 3a ($\text{score}(B) > \text{score}(A)$)**: The marginal flips $B \setminus A$ improved the public score over $A$; $\{A, B\}$ resolves to $B$ (`PROMOTE_SUPERSET_B`).
     - **Case 3b ($\text{score}(B) \le \text{score}(A)$)**: The marginal flips $B \setminus A$ failed to improve over $A$; $B \setminus A$ is rejected and $\{A, B\}$ resolves to $A$ (`RETAIN_SUBSET_A_REJECT_MARGIN`).
4. **Single Winner Already Scored Guard**:
   - If exactly 1 resolved winning block remains (`len(resolved_winners) == 1`), it was already evaluated on Kaggle as an input probe. The engine returns `status = "SINGLE_WINNER_ALREADY_SCORED"` with `output_written = False` and refuses to write a duplicate CSV for the 5th slot.
5. **Disjoint Union (`>= 2` Resolved Winners)**:
   - Combines pairwise-disjoint resolved winning blocks into a validated 30,000-row CSV, verifies the output through `submission_guard.check_submission_file`, and sets `expected_score = None` (`additive_delta_sum_prohibited = True`).

---

## 5. Summary of Executed Verification Scenarios

| Scenario ID | Scenario Description | Expected Behavior | Verified Outcome |
| :--- | :--- | :--- | :--- |
| `scenario_0` | Source AST/token audit of `build_from_scores.py` | Zero filename-number checks or hardcoded ID sets | **PASS** (`violations_found = []`) |
| `scenario_1` | Two disjoint winning blocks (15 + 20 flips) + tie + loss | Builds 35-flip union CSV; excludes tie/loss; `expected_score=None` | **PASS** (`COMBINED_DISJOINT_UNION`, 35 flips, SHA256 `491fcfa1aeb8...`) |
| `scenario_2` | Two partially overlapping winning blocks (20 and 25 sharing 10 IDs) | Raises `SubmissionBuildError` listing the 10 shared IDs; refuses `B1 U B2` and `B2 \ B1` | **PASS** (Blocked; reported shared IDs `[732, 3802, 6128, 6487, 6940, 7575, 18790, 19403, 25976, 29552]`) |
| `scenario_3` | Valid nested contrast $A$ (15) $\subset B$ (20) + disjoint $D$ (20) | Case 3a ($\text{score}(B)>\text{score}(A)$) $\to$ 40 flips; Case 3b ($\text{score}(B)\le\text{score}(A)$) $\to$ 35 flips | **PASS** (40 flips on `PROMOTE_SUPERSET_B`; 35 flips on `RETAIN_SUBSET_A_REJECT_MARGIN`) |
| `scenario_4` | Same number in filename (`candidate_sub78_v1.csv` vs `candidate_sub78_v2.csv`) with different contents | Distinguishes legacy 25-flip block (`f5090bd5...`) from disjoint 15-flip block (`96d3378b...`) purely by hash | **PASS** (`v1` blocked on 10 shared IDs; `v2` combined into 35 flips) |
| `scenario_5` | Winning superset $B$ (20 flips, `0.94650`) with losing subset $A$ (15 flips, `0.94620`) | Raises `SubmissionBuildError`; refuses to extract unprobed 5-flip subset $B \setminus A$ | **PASS** (Blocked unprobed marginal IDs `[732, 4969, 18735, 18790, 25747]`) |
| `scenario_6` | Duplicate of S48/S76, duplicate of registry entry (`75`), and single winner guard | Rejects S48/S76 and registry duplicates; returns `SINGLE_WINNER_ALREADY_SCORED` without writing CSV | **PASS** (`output_written = False`) |
| `scenario_7` | Invalid schema, labels (`"refurbished"`), duplicate/missing IDs, SHA256 mismatch, `NaN`/`Inf` scores | Rejects all invalid inputs prior to decision synthesis | **PASS** (All 6 sub-checks raised `SubmissionBuildError`) |
| `scenario_8` | Real candidate CSVs in `submissions/20261030/` | 4 disjoint blocks combine to 55 flips (`46552907...`); legacy 77+78 overlap blocked | **PASS** (55-flip disjoint union verified; legacy overlap blocked) |
