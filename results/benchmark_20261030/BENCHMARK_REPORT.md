# Benchmark Restoration, Baseline Attribution & Variant Provenance Report

**Date**: 2026-10-30  
**Repository**: `arirakowitba/tp_ap`  
**Branch**: `antigravity/overnight-recovery-coordination-20261030`  
**Active Competitor Benchmark**: Submission 48 (Score `.94633`, SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`)

---

## 1. Executive Summary: Exact Historical Reproduction (`A_hist`) vs. New Attribute-Value Variant (`A_valname`)

Our provenance audit resolved the apparent discrepancy between the earlier draft of this report (which reported 100.00% agreement with historical R151) and `reproduction_manifest_agent2.json` (which reported ~98.35% agreement):

1. **`Pipeline A_hist` (Exact Historical R151 Reproduction)**:
   - Uses `extract_deep_features` from `src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py`, where `combos` for `txt_sparse_p` concatenates **attribute IDs** (`a.get("id")`) separated by `||` (`f"{t} || {w} || {attrs}"`) and `float64` probabilities.
   - Reproduces `paired_deep_ablation_seed_2026.csv` and `paired_deep_ablation_seed_314159.csv` with **`0.000000` difference after 5-decimal rounding (`max_abs_diff = 4.9998e-6`), `100.0000%` prediction agreement, and `r = 1.000000`**.
2. **`Pipeline A_valname` (New Variant with Attribute `value_name` and Threshold Tuned on `tune`)**:
   - Uses `src/benchmark_20261030/data_loader.py`, where `combos` for `txt_sparse_p` concatenates **attribute value names** (`a.get("value_name")`) and casts `txt_sparse_p` to `float32`.
   - Produces a distinct `txt_sparse_p` feature that agrees **`98.3495%`** (Seed 2026) and **`98.2948%`** (Seed 314159) with `A_hist` (`r = 0.99637`).

| Seed | Pipeline Variant | Text `combos` Definition | Tune Threshold | Tune Acc | Confirm Acc (`th_tune`) | Confirm Acc (`0.500`) | Confirm Log Loss | Agreement vs R151 CSV | Pearson Corr vs R151 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026** | **`A_hist` (Exact R151)** | `title \|\| warranty \|\| attr.id` | `0.470` | 0.92809 | **0.92391** | 0.92437 | **0.18764** | **100.0000%** | **1.000000** |
| **2026** | **`A_valname` (New Variant)** | `title warranty attr.value_name` | `0.515` | 0.92971 | **0.92491** | 0.92445 | 0.18773 | 98.3495% | 0.996371 |
| **314159** | **`A_hist` (`6a71f6c`, `tc=8`)** | `title \|\| warranty \|\| attr.id` | `0.535` | 0.93290 | **0.92019** | 0.92089 | 0.19558 | **100.0000%** | **1.000000** |
| **314159** | **`A_hist` (`tc=4` bounded)** | `title \|\| warranty \|\| attr.id` | `0.535` | 0.93217 | **0.91996** | 0.92089 | **0.19552** | 99.5095% | 0.999625 |
| **314159** | **`A_valname` (New Variant)** | `title warranty attr.value_name` | `0.455` | 0.93209 | **0.91809** | 0.91910 | 0.19701 | 98.2948% | 0.996329 |
| **20261030** | **`A_hist` (Exact R151)** | `title \|\| warranty \|\| attr.id` | `0.440` | 0.93339 | **0.92920** | **0.93137** | **0.17039** | *(Reference)* | *(Reference)* |
| **20261030** | **`A_valname` (New Variant)** | `title warranty attr.value_name` | `0.470` | 0.93435 | **0.93028** | 0.92997 | 0.17042 | 98.7579% vs `A_hist` | 0.996864 |

- **Preserved Historical Confirm Files (Never Overwritten)**:
  - `results/benchmark_20261030/preds_A_seed2026_confirm.csv.gz` (restored from `6a71f6c`, SHA256 `0db12d52610bbff70b8992b7e24171b6cf85a67bca7dcaeb7aa87798766ab1bd`)
  - `results/benchmark_20261030/preds_A_seed314159_confirm.csv.gz` (restored from `6a71f6c`, SHA256 `9289a5be3822cd8e2e389d665e69f8d6bede7825a7bff49660fdaac06a5049f1`)
- **Explicitly Separated New Artifacts**:
  - `results/benchmark_20261030/preds_A_hist_seed{2026,314159,20261030}_{tune,confirm}.csv.gz`
  - `results/benchmark_20261030/preds_A_valname_seed{2026,314159,20261030}_{tune,confirm}.csv.gz`

---

## 2. Dissection of the Round 156/157 Baseline Drop (`.924`/`.920` -> `.905`/`.898`)

In Round 156/157, the reported CatBoost baseline fell sharply to `.905` (Seed 2026) and `.898` (Seed 314159), leading the authors to claim a "+144 net gain" for an ensemble that only achieved `.915/.911`—still far below the true `.924/.920` reference.

### Controlled Ablation Results (`results/benchmark_20261030/attribution_gap_table.csv`)

| Seed | Architecture / Pipeline | Split Partition | Features | TE $\alpha$ | Tune Acc | Confirm Acc | Confirm Log Loss | Delta vs R151 Base |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2026** | Pipeline A (R151 Base) | R151 `split_seller_groups` | R151 full deep (105 cols incl. `txt_sparse_p`) | 5.0 | 0.92809 | **0.92391** | 0.18764 | Benchmark |
| **2026** | Pipeline A (on R157 split) | R157 `canonical_split` | R151 full deep (105 cols incl. `txt_sparse_p`) | 5.0 | 0.92818 | **0.92383** | 0.19016 | **-0.00008** (1 row) |
| **2026** | Pipeline A (TE $\alpha=10$) | R151 `split_seller_groups` | R151 full deep (105 cols incl. `txt_sparse_p`) | 10.0 | 0.92758 | **0.92429** | 0.18843 | **+0.00038** |
| **314159** | Pipeline A (R151 Base) | R151 `split_seller_groups` | R151 full deep (105 cols incl. `txt_sparse_p`) | 5.0 | 0.93290 | **0.92019** | 0.19558 | Benchmark |
| **314159** | Pipeline A (on R157 split) | R157 `canonical_split` | R151 full deep (105 cols incl. `txt_sparse_p`) | 5.0 | 0.93137 | **0.92012** | 0.19749 | **-0.00007** (1 row) |
| **314159** | Pipeline A (TE $\alpha=10$) | R151 `split_seller_groups` | R151 full deep (105 cols incl. `txt_sparse_p`) | 10.0 | 0.93161 | **0.92159** | 0.19654 | **+0.00140** |

### Definitive Conclusions of the Attribution Audit
1. **The split partition DOES NOT explain the drop**: Difference is less than `0.00008` (~1 row out of 12,905).
2. **TE smoothing DOES NOT explain the drop**: Changing $\alpha=5.0$ to $\alpha=10.0$ slightly increases confirm accuracy.
3. **The entire drop was caused by feature truncation in R156/R157**, predominantly omitting `txt_sparse_p` and 9 granular visual/temporal features (`pic_code`, `thumb_code`, `pic_prefix`, `pic_mean_area`, `dur_days`, `created_dow`, `upd_minus_created`, `var_sold`, `seller_mean_init_q`).

---

## 3. Seen vs. New Seller Proportions (Validation Confirm vs. 30k Test)

- **Canonical R151 Validation `confirm` (`is_seen = np.isin(sellers[confirm], fit_sellers)`)**:
  - Seed 2026: **3,903 seen items (`30.24%`)** and **9,002 new items (`69.76%`)** (`n = 12,905`).
  - Seed 314159: **3,838 seen items (`29.88%`)** and **9,005 new items (`70.12%`)** (`n = 12,843`).
  - Seed 20261030: **3,740 seen items (`29.04%`)** and **9,141 new items (`70.96%`)** (`n = 12,881`).
- **Kaggle 30k Test Set (`is_seen = np.isin(sellers_test, sellers_train_70k)`)**:
  - **21,487 seen items (`71.623%`)** and **8,513 new items (`28.377%`)** (`n = 30,000`).

---

## 4. Confirm Overlap Audit (Never Call a Reused Confirm Split "Virgin")

- `seed_2026` vs `seed_314159`: **2,337 shared rows (`18.11%`)**, **2,089 shared sellers (`31.42%`)**.
- `seed_2026` vs `seed_20261030`: **2,393 shared rows (`18.54%`)**, **2,073 shared sellers (`31.18%`)**.
- `seed_314159` vs `seed_20261030`: **2,249 shared rows (`17.51%`)**, **2,020 shared sellers (`30.41%`)**.
- **`seed_20261030` vs union `(2026 ∪ 314159)`**: **4,232 shared rows (`32.85%`)** and **3,170 shared sellers (`48.13%`)**.
- Because these splits overlap substantially and share ~65% of `fit` rows, gains across seeds must **never** be summed or treated as independent trials.
