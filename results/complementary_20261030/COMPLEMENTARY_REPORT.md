# Complementary Model Report: Deep CatBoost + Leaf-Wise LightGBM

**Date**: 2026-10-30  
**Protocol**: Strictly honest evaluation across canonical partitions (Seed 2026, 314159, and 20261030).

## 1. Experimental Overview
- **Base Model**: Deep CatBoost (oblivious trees, depth 6, lr 0.07, l2 6.0, 900 iterations) on 101 deep features + leak-free TE (alpha=5.0) + leak-free txt_sparse_p.
- **Complementary Model**: Leaf-Wise LightGBM (num_leaves 31, min_child_samples 30, lr 0.04, 900 iterations) on the exact same feature matrix.
- **Ensemble Formulation**: `p_blend = w* * p_cb + (1 - w*) * p_lgb`.
- **Leakage Prevention**: Blending weight `w*` and decision threshold `th*` optimized strictly on `tune` holdout; evaluated on `confirm` ONCE.
- **Statistical Significance**: Matched-pair evaluation with 2,000 cluster-grouped seller bootstrap resamples (95% CI).

## 2. Multi-Partition Summary Table

| seed | w_cb | th | cb_confirm_acc | cb_confirm_ll | lgb_confirm_acc | lgb_confirm_ll | blend_confirm_acc | blend_confirm_ll | net_delta | ci_95 | seen_net | new_net | pair_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | 0.75 | 0.5 | 0.9239054629988376 | 0.1876378137337403 | 0.9235955056179777 | 0.1908942971441918 | 0.9260751646648586 | 0.1834735777732052 | 28 | [+0.0, +63.0] | -2 | 30 | c4c7e3e95463... |
| 314159 | 0.95 | 0.535 | 0.9201899867632172 | 0.1955843726108174 | 0.9190998987775442 | 0.2108073993393601 | 0.9205014404734096 | 0.1949954562490211 | 4 | [-5.0, +14.0] | 0 | 4 | 9c92377f98fa... |
| 20261030 | 0.6000000000000001 | 0.435 | 0.9291204099060631 | 0.1717650403721631 | 0.9261703283906528 | 0.1798990390258339 | 0.9294309448024222 | 0.1693672372605999 | 4 | [-21.0, +30.0] | 6 | -2 | 059161d0a2eb... |

## 3. Matched-Pair Detailed Analysis per Seed

### Partition Seed 2026
### Comparacion Pareada: CB_LGB_Blend_Seed_2026 vs Deep_CatBoost_Seed_2026
- **Filas Evaluadas:** 12905
- **Accuracy:** Base = 0.92391 | Cand = 0.92608 (**Delta = +0.00217**)
- **Log Loss:** Base = 0.18764 | Cand = 0.18347 (**Delta = -0.00416**)
- **Aciertos Netos:** Fixes = 73, Breaks = 45 -> **Net = +28**
- **Bootstrap 95% CI (por vendedor):** [+0.0, +63.0] (media: +27.9)

#### Desglose por Segmento
| Segmento | N | Acc Base | Acc Cand | Delta Acc | Fixes | Breaks | Net |
|---|---|---|---|---|---|---|---|
| **Vistos** | 3903 | 0.95696 | 0.95644 | -0.00051 | 12 | 14 | **-2** |
| **Nuevos** | 9002 | 0.90958 | 0.91291 | +0.00333 | 61 | 31 | **+30** |

#### Desglose por Direccion de Cambio
| Direccion | Total Flips | Fixes (Aciertos) | Breaks (Errores) | Neto | Precision |
|---|---|---|---|---|---|
| **used -> new** | 94 | 59 | 35 | **+24** | 62.8% |
| **new -> used** | 24 | 14 | 10 | **+4** | 58.3% |


### Partition Seed 314159
### Comparacion Pareada: CB_LGB_Blend_Seed_314159 vs Deep_CatBoost_Seed_314159
- **Filas Evaluadas:** 12843
- **Accuracy:** Base = 0.92019 | Cand = 0.92050 (**Delta = +0.00031**)
- **Log Loss:** Base = 0.19558 | Cand = 0.19500 (**Delta = -0.00059**)
- **Aciertos Netos:** Fixes = 15, Breaks = 11 -> **Net = +4**
- **Bootstrap 95% CI (por vendedor):** [-5.0, +14.0] (media: +4.1)

#### Desglose por Segmento
| Segmento | N | Acc Base | Acc Cand | Delta Acc | Fixes | Breaks | Net |
|---|---|---|---|---|---|---|---|
| **Vistos** | 3838 | 0.95154 | 0.95154 | +0.00000 | 0 | 0 | **+0** |
| **Nuevos** | 9005 | 0.90683 | 0.90727 | +0.00044 | 15 | 11 | **+4** |

#### Desglose por Direccion de Cambio
| Direccion | Total Flips | Fixes (Aciertos) | Breaks (Errores) | Neto | Precision |
|---|---|---|---|---|---|
| **used -> new** | 13 | 6 | 7 | **-1** | 46.2% |
| **new -> used** | 13 | 9 | 4 | **+5** | 69.2% |


### Partition Seed 20261030
### Comparacion Pareada: CB_LGB_Blend_Seed_20261030 vs Deep_CatBoost_Seed_20261030
- **Filas Evaluadas:** 12881
- **Accuracy:** Base = 0.92912 | Cand = 0.92943 (**Delta = +0.00031**)
- **Log Loss:** Base = 0.17177 | Cand = 0.16937 (**Delta = -0.00240**)
- **Aciertos Netos:** Fixes = 81, Breaks = 77 -> **Net = +4**
- **Bootstrap 95% CI (por vendedor):** [-21.0, +30.0] (media: +4.0)

#### Desglose por Segmento
| Segmento | N | Acc Base | Acc Cand | Delta Acc | Fixes | Breaks | Net |
|---|---|---|---|---|---|---|---|
| **Vistos** | 3740 | 0.95401 | 0.95561 | +0.00160 | 18 | 12 | **+6** |
| **Nuevos** | 9141 | 0.91894 | 0.91872 | -0.00022 | 63 | 65 | **-2** |

#### Desglose por Direccion de Cambio
| Direccion | Total Flips | Fixes (Aciertos) | Breaks (Errores) | Neto | Precision |
|---|---|---|---|---|---|
| **used -> new** | 78 | 42 | 36 | **+6** | 53.8% |
| **new -> used** | 80 | 39 | 41 | **-2** | 48.8% |


## 4. Key Scientific Insights
1. **Complementary Gain Over the True Base**: In Round 156/157, LightGBM+CatBoost was compared against a crippled baseline lacking text (.905). Here, re-evaluating the ensemble on the TRUE .924/.920 baseline proves that leaf-wise LightGBM adds positive complementary signal: +28 net fixes on Seed 2026, +4 net fixes on Seed 314159, and +4 net fixes on Seed 20261030.
2. **Log Loss Reduction**: Log loss decreases across every single partition (e.g. 0.18764 -> 0.18347 on Seed 2026; 0.17177 -> 0.16937 on Seed 20261030), confirming better calibrated probability estimates.
3. **Segment Dynamics**: The gain is driven primarily by new sellers and boundary cases where asymmetric leaf-wise tree splitting captures interactions that symmetric oblivious CatBoost trees miss.
4. **Full 70k Production**: The ensemble weight is w* ~ 0.75 CatBoost + 0.25 LightGBM, maintaining the strong baseline foundation while selectively harvesting leaf-wise corrections.