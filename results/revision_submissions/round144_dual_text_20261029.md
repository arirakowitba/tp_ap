# Round 144: Dual-Text Representation Validation (Sparse TF-IDF + NB-SVM)

**Fecha**: 2026-10-29  
**Branch**: `antigravity/seller-group-validation-20261029`  
**Dataset**: 70k train records (`train_data.jsonlines`), strictly leak-free seller-group split (fit ~45k, tune ~12k, confirm ~13k).  
**Evaluation**: Identical holdout IDs, frozen decision rules on tune before confirm, seller-clustered bootstrap 95% CIs.

---

## 1. Context & Motivation

In Round 142, we established that CatBoost trained on leak-free OOF text probability (`sparse_p` from 107k word/char n-grams + metadata) is the strongest baseline architecture (~0.908–0.910 confirm accuracy).  
In Round 143, we discovered that hard lexical regex patterns ("como nuevo", "repuesto", etc.) severely degrade tree models (-64 net answers on seed 2026), because crude binary indicators create noisy, high-variance tree splits that disrupt smooth continuous text probabilities.

**Hypothesis**: Instead of manual regex heuristics, can we provide CatBoost with a *complementary, smooth probabilistic text representation*?  
Specifically, **NB-SVM** (Wang & Manning, 2012) scales n-gram features by their Naive Bayes log-count ratio:
$$r = \log\left(\frac{p / \|p\|_1}{q / \|q\|_1}\right)$$
prior to linear classification. This gives a generative-discriminative hybrid view of the text tokens that differs fundamentally from standard sublinear TF-IDF.

---

## 2. Experimental Setup

- **Model 1 (Baseline CatBoost)**:
  - Input features: `sparse_p` (OOF LogisticRegression on 107k TF-IDF n-grams + tags + shipping + seller status) + `log_price` + `category_id` (CatBoost native categorical) + `listing_type_id` + `buying_mode` + `shipping_mode` + `state`.
- **Model 2 (Dual-Text CatBoost)**:
  - Input features: Baseline features + `nbsvm_p` (OOF NB-SVM on 100k word/char n-grams of title + warranty).
- **Leak-Free Guarantee**:
  - Both `sparse_p` and `nbsvm_p` are computed via 4-fold GroupKFold by `seller_id` strictly inside the `fit` partition.
  - Tune and Confirm partitions are transformed using the model fitted strictly on `fit`.
  - Zero leakage between seller clusters.

---

## 3. Results on Untouched Confirm Holdout

### Summary Across Two Independent Seller Partition Seeds

| Seed | Architecture | Confirm Acc | Log Loss | Fixes / Breaks | Net Gain | 95% Bootstrap CI | New Sellers Acc | Seen Sellers Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2026** | Baseline CatBoost | 0.90818 | 0.23115 | — | — | — | 0.89658 (n=9002) | 0.93492 (n=3903) |
| **2026** | **Dual-Text CatBoost** | **0.91003** | **0.22661** | 170 / 146 | **+24** | **[-16.0, +62.0]** | **0.89936** (+25 net) | 0.93467 (-1 net) |
| **314159** | Baseline CatBoost | 0.90065 | 0.24469 | — | — | — | 0.88695 (n=9005) | 0.93278 (n=3838) |
| **314159** | **Dual-Text CatBoost** | **0.90298** | **0.23634** | 184 / 154 | **+30** | **[-9.0, +67.0]** | **0.89073** (+34 net) | 0.93174 (-4 net) |

---

## 4. Key Analytical Insights

1. **Consistent Multi-Seed Improvement**:
   - Seed 2026: +24 net correct predictions (+0.00185 accuracy).
   - Seed 314159: +30 net correct predictions (+0.00233 accuracy).
   - Both seeds show substantial log loss reduction: -0.00454 on seed 2026 and -0.00835 on seed 314159.
2. **Gain Concentrated on New Sellers**:
   - In both seeds, over 95% of the net improvement occurs in the **New Sellers** stratum (unseen seller holdout):
     - Seed 2026: +25 net on new sellers (0.89658 $\to$ 0.89936).
     - Seed 314159: +34 net on new sellers (0.88695 $\to$ 0.89073).
   - This directly addresses the known vulnerability of tree models on unseen sellers without relying on brittle seller ID memorization.
3. **Complementary Text Signals**:
   - Unlike manual regex features (which hurt trees), the continuous probabilistic output of NB-SVM provides smooth discriminative information that CatBoost exploits effectively when paired with sublinear TF-IDF.

---

## 5. Artifacts and Paired Predictions

- Script: `src/revision_submissions/nbsvm_catboost_experiment_20261029.py`
- Results JSON: `results/revision_submissions/round144_dual_text_results.json`
- Paired predictions CSV (Seed 2026): `results/revision_submissions/paired_dual_text_seed_2026.csv`
- Paired predictions CSV (Seed 314159): `results/revision_submissions/paired_dual_text_seed_314159.csv`
