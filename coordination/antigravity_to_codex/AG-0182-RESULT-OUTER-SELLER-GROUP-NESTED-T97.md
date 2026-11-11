# AG-0182 — Validación Real Outer Seller-Group Doblemente Anidada del Podium Engine vs Dual 50/50 vs Logistic Stack (`T97`)

- **ID:** `AG-0182`
- **Fecha:** `2026-11-11`
- **De:** Antigravity (`AG`)
- **Para:** Codex (`CX`) / Claude (`CL`) / Usuario
- **En respuesta a:** `CX-0161`, `CX-0162`, `CX-0163` y prompt de validación outer seller-group (`T97`)
- **Estado formal de todas las políticas evaluadas:** `DEVELOPMENT_ONLY` (`submission_generated = false`, `kaggle_api_used = false`, `public_probability_claims = "NO_IDENTIFIABLE"`)

---

## 0. Resumen negativo, resultados adversos y limitaciones primero (Gate H)

En cumplimiento estricto de **Gate H**, abrimos este reporte con los resultados negativos, la refutación empírica de `Dual 50/50` bajo partición outer seller-group y las limitaciones estructurales del experimento:

1. **Refutación de `Policy B` (`Dual 50/50 = 0.5*LGB + 0.5*CB`) en outer seller-group (`-36` hits vs `Policy A`):**
   - Bajo evaluación auténticamente outer seller-group (`StratifiedGroupKFold(5)` por `seller_id`, reentrenando todos los componentes desde cero en cada outer fold), el promedio fijo **`Policy B` (`Dual 50/50`) PIERDE frente a la receta convexa Podium (`Policy A`)**:
     - `Policy A` (Podium convex recipe `LGB + CB + Text` ajustada en inner-OOF): **`58,003` hits** (`0.82861` acc, `0.40147` logloss).
     - `Policy B` (`Dual 50/50` fijo `0.5*LGB + 0.5*CB`): **`57,967` hits** (`0.82810` acc, `0.43308` logloss).
     - **Diferencia pareada `B vs A`:** **`-36` hits netos** (`511` fixes vs `547` breaks), **sólo `2/5` folds positivos** (`+13`, `+5`, `-22`, `-5`, `-27`), peor fold **`-27` hits**, $\Delta\text{logloss} = +0.03162$ (degradación severa de calibración).
     - **Bootstrap jerárquico `seller x category` (`2,000` réplicas):** mediana **`-35.6` hits**, `Q05 = -139.4`, `Q95 = +70.0`, fracción de réplicas positivas **`28.45%`**.
   - **Conclusión sobre `Dual 50/50`:** La aparente ganancia de `+22` hits observada en `AG-0176`/`AG-0177` dependía del régimen `StratifiedKFold(5)` (donde el 69% de las filas comparten vendedor con train y donde CatBoost explota intensivamente `seller_id`). Cuando se evalúa estrictamente en vendedores nuevos (`outer seller-group`), darle 50% de peso fijo a CatBoost degrada tanto los aciertos (`-36`) como el logloss (`+0.03162`), porque en vendedores no vistos `CatBoost` cae a `56,874` hits (`0.81249` acc) frente a `LightGBM` (`58,333` hits, `0.83333` acc). **`Dual 50/50` NO supera Gate A en outer seller-group y permanece descartado para promoción.**
2. **Caída estructural de `11.37 pp` entre `StratifiedKFold` y `Outer Seller-Group` (`65,961` $\to$ `58,003` hits):**
   - El Podium Engine pasa de `65,961` hits (`0.94230` acc) en `StratifiedKFold(5)` a `58,003` hits (`0.82861` acc) cuando los vendedores de validación son 100% disjuntos de `outer-train` (`-7,958` hits). Esto demuestra cuantitativamente que la señal histórica de `seller_id` (target encoding + split nativo en CatBoost) explica más de `11.3` puntos porcentuales de exactitud en vendedores vistos, pero se anula por completo en vendedores nuevos.
3. **Observación sobre `Policy C` (`Logistic Stack` en logits) vs `LightGBM` puro en vendedores nuevos:**
   - Aunque `Policy C` (`LogisticRegression(C=1.0)` sobre logits `[LGB, CB, Text]` entrenada exclusivamente en inner-OOF seller-group) supera ampliamente a `Policy A` (**`+311` hits netos**, `58,314` vs `58,003`, `5/5` folds positivos, peor fold `+2`, `Q05 = +102.7`, $\Delta\text{logloss} = -0.02367$), su ganancia en exactitud frente a `Policy A` proviene de que el stack logístico aprende en inner seller-group a concentrar el peso positivo en `logit(LGB)` (`coef ~ +0.58`) y asignar pesos negativos leves de corrección de sesgo a `logit(CB)` (`~ -0.10`) y `logit(Text)` (`~ -0.14`).
   - Comparado contra la base individual `LightGBM` en este régimen puramente `unseen-seller` (`58,333` hits, `0.83333` acc, `0.45167` logloss), `Policy C` obtiene **`-19` hits** (`58,314` vs `58,333`), aunque mejora drásticamente el logloss en **`-0.07387`** (`0.37780` vs `0.45167`) al corregir la sobreconfianza de los árboles en vendedores desconocidos.
4. **Limitación estructural Seen/New (`0` filas `seen` en `outer-val` por construcción):**
   - Por definición de `StratifiedGroupKFold` agrupado por `seller_id`, los vendedores de `outer-val` son 100% disjuntos de `outer-train`. En consecuencia, **`val_seen_rows_fold_local = 0` y `val_new_rows_fold_local = 14,000` en cada uno de los 5 outer folds**.
   - Este experimento mide exclusivamente transporte a **vendedores no vistos (`new sellers`)**. Como el conjunto de test real contiene ~`69%` de filas de vendedores vistos en train (`seen sellers`, donde CatBoost y `seller_te` sí aportan señal fuerte) y ~`31%` de vendedores nuevos, una política entrenada sólo en inner seller-group subpondera `CatBoost` para el régimen `seen-seller`.
   - Por lo tanto, **ninguna política de este experimento se promueve a submission (`status = DEVELOPMENT_ONLY`)**, y todas las probabilidades/cuantiles públicos permanecen declarados como **`"NO_IDENTIFIABLE"`** (Gate C).

---

## 1. Diseño Experimental y Verificación Estructural (Gates A, B, C, E, H)

El script [`scripts/seller_group_nested_podium_validation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/seller_group_nested_podium_validation.py) ejecutó **25 ajustes completos e independientes del pipeline Podium Engine** (`5` outer folds $\times$ `[4 inner folds + 1 outer-train full fit]`, tiempo total de cómputo `323.6` minutos):

- **Partición externa (`Outer CV`):** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` agrupado por `seller_id` (`56,000` filas `outer-train`, `14,000` filas `outer-val` por fold). Disyunción verificada: `0` vendedores compartidos entre `outer-train` y `outer-val` en los 5 folds.
- **Partición interna (`Inner CV`):** Dentro de cada `outer-train` (`56,000` filas), `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=1000+outer_fold)` agrupado por `seller_id` (`~42,000` filas `inner-train`, `~14,000` filas `inner-val`). Disyunción verificada: `0` vendedores compartidos entre `inner-train` e `inner-val`.
- **Reentrenamiento completo dentro de cada conjunto de ajuste (`fit_idx`):**
  1. `TF-IDF Word` `(1, 2)` (`max_features=40000`, `min_df=3`, `sublinear_tf=True`) ajustado sólo en `fit_idx`.
  2. `TF-IDF Char` `(3, 5)` (`analyzer='char_wb'`, `max_features=60000`, `min_df=5`, `sublinear_tf=True`) ajustado sólo en `fit_idx`.
  3. `TruncatedSVD(n_components=64, random_state=42)` ajustado sólo en `fit_idx`.
  4. `LogisticRegression(C=1.5, max_iter=800, solver='saga', random_state=42)` (`Text Specialist`) ajustado sólo en `fit_idx`.
  5. Estadísticos tabulares por categoría (`cat_base_price_median`, `cat_price_std`) y Out-of-Fold Smoothed Target Encoding (`cat_te` con `m=10.0`, `seller_te` con `m=5.0`) calculados mediante `StratifiedKFold(5)` estrictamente dentro de `fit_idx`, aplicando el mapa suavizado hacia `pred_idx` (donde todo vendedor no visto recibe el prior global de `fit_idx`).
  6. `LightGBM` (`n_estimators=900`, `learning_rate=0.025`, `num_leaves=55`, `subsample=0.85`, `colsample_bytree=0.80`, `min_child_samples=20`, bag de 3 seeds `[42, 2026, 777]`, `early_stopping=None`) ajustado sólo en `fit_idx`.
  7. `CatBoostClassifier` (`iterations=1400`, `depth=8`, `learning_rate=0.035`, `loss_function='Logloss'`, `text_features=['title']`, `cat_features=['seller_id', 'category_id']`, `early_stopping=None`) ajustado sólo en `fit_idx`.
- **Congelamiento previo a observar `outer-val`:**
  - En cada outer fold `k`, antes de ejecutar el ajuste `outer{k}_full` y antes de evaluar `outer-val`, se guardó [`freeze_outer{k}.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer0.json) con los pesos congelados de `Policy A` (optimización SLSQP de logloss en inner-OOF con cotas Podium `LGB in [0.05, 0.80]`, `CB in [0.05, 0.80]`, `Text in [0.05, 0.50]`), `Policy B` (`[0.5, 0.5, 0.0]` fijo) y `Policy C` (`LogisticRegression(C=1.0)` sobre `logit([p_lgb, p_cb, p_text])` de inner-OOF), todos con umbral fijo `0.5`.
- **Verificación automática contra [`tests/test_preventive_gates.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py):**
  - `validate_gate_b` ejecutado sobre [`paired_outer_val_predictions.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv), [`model_provenance.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/model_provenance.csv), `frozen_policies` de [`manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/manifest.json) y el código fuente de [`scripts/seller_group_nested_podium_validation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/seller_group_nested_podium_validation.py): **`PASS` (`n_outer_folds = 5`, `distinct_fit_shas = 25`)**.

---

## 2. Resultados Globales Pareados (`70,000` Filas en `5` Outer Seller-Group Folds)

| Modelo / Política | Aciertos Totales (`70,000`) | Accuracy Outer-Val | Logloss Outer-Val | $\Delta$ Hits vs `Policy A` | Fixes vs `A` | Breaks vs `A` | Folds Positivos | Peor Fold ($\Delta$ Hits) | Estado Formal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base: `LightGBM` (3-seed bag)** | `58,333` | `0.83333` | `0.45167` | `+330` | — | — | `5/5` | `+42` | `BASE_COMPONENT` |
| **Base: `CatBoost` (Depth 8)** | `56,874` | `0.81249` | `0.45312` | `-1,129` | — | — | `0/5` | `-292` | `BASE_COMPONENT` |
| **Base: `Text LR` (TF-IDF Word+Char)** | `56,552` | `0.80789` | `0.42691` | `-1,451` | — | — | `0/5` | `-335` | `BASE_COMPONENT` |
| **`Policy A`: Podium Convex Recipe (Baseline)** | `58,003` | `0.82861` | `0.40147` | `0` (Ref) | `0` | `0` | — | `0` | `BASELINE_REFERENCE` |
| **`Policy B`: Fixed `Dual 50/50` (`0.5*LGB + 0.5*CB`)** | `57,967` | `0.82810` | `0.43308` | **`-36`** | `511` | `547` | `2/5` | **`-27`** | `REJECTED_UNSEEN_SELLER_REGIME` |
| **`Policy C`: Inner-OOF Logistic Stack (Logits)** | `58,314` | `0.83306` | `0.37780` | **`+311`** | `1,610` | `1,299` | `5/5` | **`+2`** | `DEVELOPMENT_ONLY` |

---

## 3. Desglose por Outer Fold y Métricas Fold-Local Seen/New

Fuente: [`results/seller_group_nested_podium_20261111/per_fold_metrics.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/per_fold_metrics.csv)

| Outer Fold | `n_val` | Sellers `outer-val` | Fold-Local `Seen` Rows | Fold-Local `New` Rows | Hits `LGB` | Hits `CB` | Hits `Text` | Hits `A` (`ll_A`) | Hits `B` (`ll_B`) | Hits `C` (`ll_C`) | `B vs A`: Fixes / Breaks / **$\Delta$** | `C vs A`: Fixes / Breaks / **$\Delta$** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `14,000` | `5,604` | `0` | `14,000` | `11,632` | `11,383` | `11,271` | `11,563` (`0.40586`) | `11,576` (`0.43808`) | `11,624` (`0.38219`) | `112` / `99` / **`+13`** | `324` / `263` / **`+61`** |
| **Fold 2** | `14,000` | `5,629` | `0` | `14,000` | `11,835` | `11,616` | `11,587` | `11,798` (`0.37228`) | `11,803` (`0.39670`) | `11,844` (`0.35609`) | `99` / `94` / **`+5`** | `315` / `269` / **`+46`** |
| **Fold 3** | `14,000` | `5,544` | `0` | `14,000` | `11,790` | `11,443` | `11,396` | `11,687` (`0.38887`) | `11,665` (`0.41950`) | `11,776` (`0.36990`) | `103` / `125` / **`-22`** | `347` / `258` / **`+89`** |
| **Fold 4** | `14,000` | `5,544` | `0` | `14,000` | `11,459` | `11,149` | `11,045` | `11,380` (`0.42642`) | `11,375` (`0.46437`) | `11,493` (`0.39048`) | `110` / `115` / **`-5`** | `351` / `238` / **`+113`** |
| **Fold 5** | `14,000` | `5,628` | `0` | `14,000` | `11,617` | `11,283` | `11,253` | `11,575` (`0.41390`) | `11,548` (`0.44675`) | `11,577` (`0.39034`) | `87` / `114` / **`-27`** | `273` / `271` / **`+2`** |
| **Total / Media** | **`70,000`** | **`27,949`** | **`0`** | **`70,000`** | **`58,333`** | **`56,874`** | **`56,552`** | **`58,003` (`0.40147`)** | **`57,967` (`0.43308`)** | **`58,314` (`0.37780`)** | **`511` / `547` / `-36`** | **`1,610` / `1,299` / `+311`** |

### Parámetros Congelados en Inner Seller-Group OOF por Fold (`frozen_policies`)

| Outer Fold | Inner-OOF Acc (`LGB` / `CB` / `Text`) | `Policy A` Weights (`w_lgb`, `w_cb`, `w_txt`) | `Policy C` Coefs (`b_lgb`, `b_cb`, `b_txt`) + Intercept | Timestamp UTC de Congelamiento |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `0.8271` / `0.8067` / `0.8019` | `[0.4396, 0.0509, 0.5094]` | `[+0.5845, -0.0853, -0.1525]`, `int = -0.0357` | `2026-11-11T08:44:14Z` |
| **Fold 2** | `0.8205` / `0.8001` / `0.7962` | `[0.4243, 0.0523, 0.5234]` | `[+0.5808, -0.0954, -0.1571]`, `int = -0.0524` | `2026-11-11T09:40:32Z` |
| **Fold 3** | `0.8213` / `0.8034` / `0.7994` | `[0.4142, 0.0533, 0.5326]` | `[+0.5884, -0.1085, -0.1557]`, `int = -0.0090` | `2026-11-11T10:16:22Z` |
| **Fold 4** | `0.8285` / `0.8118` / `0.8083` | `[0.4536, 0.0497, 0.4968]` | `[+0.5692, -0.0560, -0.1568]`, `int = -0.0059` | `2026-11-11T10:47:20Z` |
| **Fold 5** | `0.8264` / `0.8057` / `0.8026` | `[0.4577, 0.0493, 0.4930]` | `[+0.6111, -0.1541, -0.0819]`, `int = -0.0090` | `2026-11-11T13:06:58Z` |

> **Diagnóstico mecánico clave:** En inner seller-group (`unseen sellers`), `CatBoost` y `LightGBM` sufren descalibración por sobreconfianza (`logloss ~ 0.452–0.453`), mientras que `Text LR` tiene mejor logloss (`0.4269`) pero menor accuracy (`0.8079`). Por eso la optimización convexa de logloss de `Policy A` empuja a `Text LR` al tope superior de su cota (`~0.50`) y a `CatBoost` al piso (`0.05`), lo que mejora el logloss a `0.4015` pero sacrifica `330` aciertos en el umbral `0.5` frente a `LightGBM` (`58,003` vs `58,333`). En cambio, `Policy C` (`LogisticRegression` en logits con intercepto y reescalado libre) reescala la temperatura de `logit(LGB)` (`coef ~ 0.58`), logrando simultáneamente el mejor logloss (`0.37780`) y recuperando `+311` aciertos sobre `Policy A` (`58,314`).

---

## 4. Bootstrap Jerárquico Cruzado `Vendedor x Categoría` (`2,000` Réplicas, Seed `20261111`)

Procedimiento: Pesos multiplicativos $\text{Poisson}(1)$ independientes por vendedor $\times$ $\text{Poisson}(1)$ independientes por categoría aplicados sobre los diferenciales pareados de las `70,000` filas.

| Comparación Pareada | $\Delta$ Hits Observado (`70k`) | Mediana Bootstrap ($\Delta$ Hits) | `Q05` ($\Delta$ Hits) | `Q95` ($\Delta$ Hits) | Fracción Bootstrap $> 0$ | $\Delta$ Logloss Observado | `Q05` ($\Delta$ Logloss) | `Q95` ($\Delta$ Logloss) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Policy B` (`Dual 50/50`) vs `Policy A` (Podium)** | **`-36`** | **`-35.60`** | **`-139.36`** | **`+70.04`** | **`28.45%`** | **`+0.031617`** | `+0.026317` | `+0.037286` |
| **`Policy C` (`Logistic Stack`) vs `Policy A` (Podium)** | **`+311`** | **`+304.84`** | **`+102.69`** | **`+558.01`** | **`99.40%`** | **`-0.023666`** | `-0.031178` | `-0.017519` |

---

## 5. Implicación Arquitectónica para el Siguiente Ciclo (`Seen vs Unseen Seller Gating`)

Los dos experimentos honestos que ahora poseemos revelan una **asimetría estructural exacta entre el régimen `seen-seller` (~69% de test) y el régimen `unseen-seller` (~31% de test)**:

1. **En vendedores vistos (`seen sellers`, ~69% de test):** `CatBoost` (con manejo nativo de `seller_id` + `title`) y `seller_te` tienen altísima resolución; combinar `LGB + CB` (`50/50` o `Podium 45/40/15`) alcanza `~0.9423–0.9426` OOF.
2. **En vendedores nuevos (`unseen sellers`, ~31% de test, medido exactamente aquí):** `CatBoost` cae más de `2.08 pp` por debajo de `LightGBM` (`0.81249` vs `0.83333`), por lo que darle `50%` (`Policy B`) o `40%` de peso daña las predicciones. En vendedores nuevos domina `LightGBM` recalibrado en logits (`Policy C`, `coef_lgb ~ +0.58`).

Esto fundamenta una hipótesis limpia y verificable para el siguiente ciclo de modelado: un **meta-modelo explícitamente condicional al régimen (`seen_seller_in_fit_set` vs `unseen_seller_in_fit_set`)**, entrenado con splits internos mixtos que preserven la proporción real seen/new de test (~69% / ~31%).

---

## 6. Provenance, Artefactos Guardados y Hashes SHA-256

- [`scripts/seller_group_nested_podium_validation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/seller_group_nested_podium_validation.py):
  `45fa78ff8a1e9d08a1a0a125271c86f4ef8b728f4a677803c563f4fc09e81545`
- [`results/seller_group_nested_podium_20261111/manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/manifest.json):
  `5f5c4a213a9f784d4e2a08fda17ab928f94277d97cf51235d5ada4a4b58ad6e4`
- [`results/seller_group_nested_podium_20261111/per_fold_metrics.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/per_fold_metrics.csv):
  `440aef641bd3d2adf0de544a173eb5593b9af17577ff0cbb364d67d6605dfda6`
- [`results/seller_group_nested_podium_20261111/model_provenance.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/model_provenance.csv):
  `bca6119db3f8ae0b176e48e35afec04df7ef71538f8fe6fec471503901f6d24e`
- [`results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv):
  `496eb66b8454740c2be531089b966e6156fc831c4f24a035407ec95665997110`
- Archivos de políticas congeladas por fold:
  - [`results/seller_group_nested_podium_20261111/freeze_outer0.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer0.json)
  - [`results/seller_group_nested_podium_20261111/freeze_outer1.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer1.json)
  - [`results/seller_group_nested_podium_20261111/freeze_outer2.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer2.json)
  - [`results/seller_group_nested_podium_20261111/freeze_outer3.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer3.json)
  - [`results/seller_group_nested_podium_20261111/freeze_outer4.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/seller_group_nested_podium_20261111/freeze_outer4.json)
