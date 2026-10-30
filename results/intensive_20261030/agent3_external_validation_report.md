# Agente 3 — Validación Externa al Proceso de Selección y Transporte a la Mezcla de Test (Revisión CX-0002)

**Fecha:** 2026-10-30 02:21:53  
**Tiempo total de ejecución:** 2020.65s  
**Clasificación Epistemológica:** `DEVELOPMENT_GRADE_PROMISING_LOGLOSS_MODERATE_ACCURACY`  

> [!WARNING]
> Audited in all_historical_holdouts_overlap_audit.json: the union of all 18 historical holdouts covers 67,007 of 70,000 train rows (95.72%). Although `disjoint_3seed_20261101` was constructed to have 0.0% overlap with the recent confirm sets of 2026, 314159, and 20261030, it overlaps in 53.49% with their tune+confirm union and in 92.39% (8,862 / 9,592) with the union of all 18 historical holdouts. Sharing fit rows across experiments does not constitute supervised leakage, but does rule out sample independence. All 3 cuts execute strictly inductive per-fold vocabulary and TF-IDF refit inside `fit`.

---

## 1. Tamaños, Hashes y Solapamiento Histórico de los Cortes Externos Inductivos

| Corte Externo | Semilla | `N_fit` | `N_tune` | `N_confirm` | `Seen` Confirm | `New` Confirm | Solape 18 Holdouts Históricos | SHA256 Paired CSV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | `20261101` | 47,157 | 13,251 | 9,592 | 4,202 (43.81%) | 5,390 (56.19%) | `92.39% (8,862 filas)` | `42d893124e48...` |
| **`nested_outer_777001`** | `777001` | 44,415 | 12,782 | 12,803 | 3,836 (29.96%) | 8,967 (70.04%) | `95.63% (12,243 filas)` | `5249b16ecdb6...` |
| **`nested_outer_777002`** | `777002` | 44,158 | 12,583 | 13,259 | 3,815 (28.77%) | 9,444 (71.23%) | `96.25% (12,762 filas)` | `7ace3dee8452...` |

---

## 2. Comparación Exacta: Single Blend vs. Bag3 Blend y Aporte Marginal de Bag3

### A. Blend Bag3 vs. Base CatBoost (Configuración de Option A en Cortes Externos)

| Corte Externo | Régimen Umbral | Acc Base | Acc Bag3 | Fixes | Breaks | Neto Obs | IC 95% Vendedor Obs | Seen Net | New Net | $\Delta$ Acc Transp | Neto 30k Test | IC 95% 30k Test | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | Común 0.500 | 0.92598 | 0.92598 | 56 | 56 | **+0** | [-21.0, +21.0] | +0 | +0 | `+0.00000` | **+0.0** | [-65.6, +64.8] | `-0.00385` |
| **`disjoint_3seed_20261101`** | Política tune | 0.92462 | 0.92619 | 75 | 60 | **+15** | [-8.0, +38.0] | -3 | +18 | `+0.00044` | **+13.1** | [-59.1, +85.2] | `-0.00385` |
| **`nested_outer_777001`** | Común 0.500 | 0.92689 | 0.92760 | 64 | 55 | **+9** | [-13.0, +32.0] | +11 | -2 | `+0.00199` | **+59.7** | [+6.3, +116.5] | `-0.00257` |
| **`nested_outer_777001`** | Política tune | 0.92705 | 0.92736 | 65 | 61 | **+4** | [-21.0, +29.0] | +10 | -6 | `+0.00168` | **+50.3** | [-2.3, +101.8] | `-0.00257` |
| **`nested_outer_777002`** | Común 0.500 | 0.92533 | 0.92805 | 99 | 63 | **+36** | [+10.0, +63.0] | +7 | +29 | `+0.00219` | **+65.6** | [-1.9, +134.1] | `-0.00414` |
| **`nested_outer_777002`** | Política tune | 0.92398 | 0.92828 | 109 | 52 | **+57** | [+30.0, +87.0] | +9 | +48 | `+0.00313` | **+94.0** | [+33.9, +156.9] | `-0.00414` |

### B. Single Blend vs. Base CatBoost (Un solo modelo por arquitectura)

| Corte Externo | Régimen Umbral | Acc Base | Acc Single | Fixes | Breaks | Neto Obs | IC 95% Vendedor Obs | Seen Net | New Net | $\Delta$ Acc Transp | Neto 30k Test | IC 95% 30k Test | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | Común 0.500 | 0.92598 | 0.92577 | 38 | 40 | **-2** | [-19.0, +15.0] | -2 | +0 | `-0.00034` | **-10.2** | [-64.3, +44.6] | `-0.00282` |
| **`disjoint_3seed_20261101`** | Política tune | 0.92462 | 0.92473 | 90 | 89 | **+1** | [-25.0, +28.0] | -2 | +3 | `-0.00018` | **-5.5** | [-81.3, +71.2] | `-0.00282` |
| **`nested_outer_777001`** | Común 0.500 | 0.92689 | 0.92697 | 48 | 47 | **+1** | [-19.0, +21.0] | +8 | -7 | `+0.00127` | **+38.2** | [-12.5, +89.8] | `-0.00253` |
| **`nested_outer_777001`** | Política tune | 0.92705 | 0.92744 | 79 | 74 | **+5** | [-23.0, +32.0] | +7 | -2 | `+0.00124` | **+37.3** | [-16.5, +88.4] | `-0.00253` |
| **`nested_outer_777002`** | Común 0.500 | 0.92533 | 0.92631 | 53 | 40 | **+13** | [-6.0, +33.0] | +4 | +9 | `+0.00102` | **+30.6** | [-30.4, +88.6] | `-0.00269` |
| **`nested_outer_777002`** | Política tune | 0.92398 | 0.92767 | 90 | 41 | **+49** | [+24.0, +76.0] | +5 | +44 | `+0.00226` | **+67.8** | [+17.4, +122.6] | `-0.00269` |

### C. Aporte Marginal Directo: Blend Bag3 vs. Single Blend

| Corte Externo | Régimen Umbral | Acc Single | Acc Bag3 | Fixes | Breaks | Neto Obs | IC 95% Vendedor Obs | Seen Net | New Net | $\Delta$ Acc Transp | Neto 30k Test | IC 95% 30k Test | $\Delta$ Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`disjoint_3seed_20261101`** | Común 0.500 | 0.92577 | 0.92598 | 28 | 26 | **+2** | [-13.0, +17.0] | +2 | +0 | `+0.00034` | **+10.2** | [-38.6, +54.4] | `-0.00103` |
| **`disjoint_3seed_20261101`** | Política tune | 0.92473 | 0.92619 | 63 | 49 | **+14** | [-6.0, +35.0] | -1 | +15 | `+0.00062` | **+18.6** | [-47.1, +79.2] | `-0.00103` |
| **`nested_outer_777001`** | Común 0.500 | 0.92697 | 0.92760 | 46 | 38 | **+8** | [-10.0, +26.0] | +3 | +5 | `+0.00072` | **+21.6** | [-24.5, +66.5] | `-0.00005` |
| **`nested_outer_777001`** | Política tune | 0.92744 | 0.92736 | 48 | 49 | **-1** | [-23.0, +20.0] | +3 | -4 | `+0.00043` | **+13.0** | [-34.4, +59.2] | `-0.00005` |
| **`nested_outer_777002`** | Común 0.500 | 0.92631 | 0.92805 | 59 | 36 | **+23** | [+3.0, +43.0] | +3 | +20 | `+0.00116` | **+34.9** | [-9.4, +77.9] | `-0.00145` |
| **`nested_outer_777002`** | Política tune | 0.92767 | 0.92828 | 39 | 31 | **+8** | [-9.0, +24.0] | +4 | +4 | `+0.00087` | **+26.1** | [-9.3, +65.3] | `-0.00145` |

---

## 3. Supuestos Explícitos de Transportabilidad

- 1. Conditional invariance within strata: E[Delta Acc | seen/new, test] == E[Delta Acc | seen/new, confirm].
- 2. Mixture reweighting: Validation confirm has ~30.2% seen / ~69.8% new rows, whereas Kaggle 30k test has 71.623% seen (21,487) / 28.377% new (8,513) rows.
- 3. Sample size scaling: Full 70k refit has 58% more training rows than the ~44.2k `fit` split, strengthening seen-seller target encodings.
- 4. Threshold sensitivity: Thresholds tuned on a 30/70 seen/new validation mixture can suffer prior-shift degradation on a 72/28 test mixture; common-threshold (0.500) comparisons isolate pure probability ranking improvements.

## 4. Análisis de Regresiones de Bag3 por Banda de Margen (`|p_base - 0.50|`)

| Corte Externo | Banda de Margen `|p_base - 0.50|` | Filas | Fixes | Breaks | Neto |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `disjoint_3seed_20261101` | `0.00-0.02 (boundary)` | 72 | 20 | 17 | **+3** |
| `disjoint_3seed_20261101` | `0.02-0.05 (narrow)` | 112 | 19 | 12 | **+7** |
| `disjoint_3seed_20261101` | `0.05-0.10 (moderate)` | 219 | 11 | 21 | **-10** |
| `disjoint_3seed_20261101` | `0.10-0.20 (wide)` | 456 | 6 | 6 | **+0** |
| `disjoint_3seed_20261101` | `0.20+ (high confidence)` | 8,733 | 0 | 0 | **+0** |
| `nested_outer_777001` | `0.00-0.02 (boundary)` | 104 | 21 | 19 | **+2** |
| `nested_outer_777001` | `0.02-0.05 (narrow)` | 179 | 23 | 22 | **+1** |
| `nested_outer_777001` | `0.05-0.10 (moderate)` | 261 | 12 | 9 | **+3** |
| `nested_outer_777001` | `0.10-0.20 (wide)` | 581 | 8 | 5 | **+3** |
| `nested_outer_777001` | `0.20+ (high confidence)` | 11,678 | 0 | 0 | **+0** |
| `nested_outer_777002` | `0.00-0.02 (boundary)` | 103 | 28 | 19 | **+9** |
| `nested_outer_777002` | `0.02-0.05 (narrow)` | 146 | 29 | 21 | **+8** |
| `nested_outer_777002` | `0.05-0.10 (moderate)` | 285 | 30 | 16 | **+14** |
| `nested_outer_777002` | `0.10-0.20 (wide)` | 659 | 11 | 7 | **+4** |
| `nested_outer_777002` | `0.20+ (high confidence)` | 12,066 | 1 | 0 | **+1** |
