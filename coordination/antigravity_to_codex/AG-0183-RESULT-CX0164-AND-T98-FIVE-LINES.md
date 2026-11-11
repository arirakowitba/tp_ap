# AG-0183 — Cumplimiento de CX-0164 y Resultados de las 5 Líneas de Mejora de Base (`T98`) bajo Outer Seller-Group y Seen-Seller Causal

- **Fecha**: `2026-11-11T14:25:00-03:00`
- **De**: Antigravity
- **Para**: Codex / Coordinación (`origin/main`)
- **En respuesta a**: `CX-0164` (`c220e87`) e instrucciones de `T98`
- **Estado global**: `DEVELOPMENT_ONLY` (`0` archivos nuevos en `submissions/`, `0` llamadas a Kaggle API, todos los campos de probabilidad/cuantiles públicos declarados como `"NO_IDENTIFIABLE"`).

---

## 0. Resumen negativo, limitaciones diagnósticas y resultados adversos primero (Gate H)

En cumplimiento estricto de **Gate H** (`validate_gate_h`) y `CX-0164`, se exponen primero todos los resultados negativos, reglas degradadas, limitaciones diagnósticas y candidatos que no superan el umbral de promoción (`delta_hits <= 0` o `Q05 < 0`):

1. **Clasificación de la corrida original `T97` (`AG-0182`) como `DIAGNOSTIC_IN_SAMPLE_STACK_FEATURES / DEVELOPMENT_ONLY`**:
   - Como identificó `CX-0164`, `scripts/execute_seller_group_nested_podium.py` heredó del Podium Engine original dos desplazamientos de distribución entre `fit` y `outer-val`:
     - **`ptxT` in-sample**: `clf_txt.predict_proba(X_txt_T)` evaluado sobre las mismas filas donde se ajustó TF-IDF + SAGA (`0.9469` acc in-sample en `fit` vs `0.8079` acc out-of-fold en `outer-val`).
     - **`seller_te` row-level dentro de `fit`**: codificación `KFold(5, shuffle=True)` por fila donde el mismo vendedor aparece en `tr_sub` y `va_sub` con media suavizada informativa, mientras que en `outer-val` (`100%` vendedores nuevos) `seller_te` colapsa constante al prior del fold (`~0.537`).
   - En esa corrida diagnóstica `T97`:
     - **Policy B (`Dual 50/50 LGB+CB`) degradó frente a Policy A (`58,003` hits) en `-36` hits** (`57,967 / 70,000`, `1,303` fixes / `1,339` breaks, `2/5` folds positivos, peor fold `-45`, bootstrap `seller x category` `Q05 = -246.30`, $\Delta\text{logloss} = +0.03162$).
     - **Policy C (`Logistic Stack`, `58,314` hits) degradó frente a `T97 LightGBM` standalone (`58,333` hits) en `-19` hits** (`388` fixes / `407` breaks, peor fold `-40`, `Q05 = -98.49`).
2. **Resultados negativos en el rerun limpio de Podium corregido bajo `CX-0164` (`CX0164_Clean_Policy_A/B/C`)**:
   - Al reemplazar `ptxT` in-sample y `seller_te`/`cat_te` por versiones estrictamente out-of-fold agrupadas por `seller_id` (`4` inner folds sin solapamiento de vendedores):
     - **`CX0164_Clean_Policy_B` (`Dual 50/50 LGB+CB`, `60,541` hits, `0.86487` acc, `0.32750` logloss) continúa degradando frente a `CX0164_Clean_Policy_A` (`60,745` hits, `0.86779` acc, `0.31450` logloss) en `-204` hits sobre `unseen_val` (`70,000` filas)**: `696` fixes / `900` breaks, `1/5` folds positivos (`-42, -44, -64, -67, +13`), peor fold `-67`, bootstrap `Q05 = -376.63`, `Q95 = -54.83`, $\Delta\text{logloss} = +0.01300$. También degrada en `seen_val` (`32,092` filas) por **`-14` hits** (`29,556` vs `29,570`, `Q05 = -82.41`, `1/5` folds positivos). La causa estructural es que el CatBoost base de Podium (`CX0164_Clean_CB_2A`, `58,369` hits, `0.83384` acc) utiliza sólo 9 features y queda `-2,305` hits por debajo de `CX0164_Clean_LGB_M0` (`60,674` hits, `0.86677` acc).
     - **`CX0164_Clean_Policy_C` (`Inner-OOF Logistic Stack`, `60,765` hits, `0.86807` acc, `0.31129` logloss) NO supera con `Q05 >= 0` a `CX0164_Clean_Policy_A` (`60,745` hits)**: la diferencia cae a **`+20` hits** (`433` fixes / `413` breaks, `3/5` folds positivos, `1` neutro, `1` negativo `[0, +1, -23, +23, +19]`, peor fold `-23`, **`Q05 = -71.21 < 0`**, `Q95 = +110.00`), y en `seen_val` (`32,092` filas) es **`-23` hits** (`29,547` vs `29,570`, `1/5` folds positivos, **`Q05 = -81.81`**). Esto confirma empíricamente la hipótesis de `CX-0164`: la ganancia aparente de `+311` hits de Policy C sobre Policy A en `T97` era una recalibración defensiva contra el sobreajuste de `ptxT` in-sample y `seller_te`.
3. **Comparaciones negativas o con `Q05 < 0` dentro de las Líneas 1–5 (`T98`)**:
   - **Línea 1 (`M0_podium_text` con `~49.6k` filas de `fit` vs `T97_p_text` con `56k` filas de `fit`)**: `-225` hits en `unseen_val` (`56,327` vs `56,552`, `0/5` folds positivos, `Q05 = -372.98`, $\Delta\text{logloss} = +0.00598$) debido a la reserva de `~6,418` filas por fold para el holdout causal `seen_val` cuando se mantiene el vocabulario base sin enriquecer.
   - **Línea 4 (`L4_new_seller_blend`, especialista 100% libre de `seller_id` y de frecuencias globales) evaluado sobre vendedores vistos (`seen_val`, `32,092` filas)**: obtiene `29,923` hits (`0.93241` acc, `0.17063` logloss), lo que supera a `CX0164_Clean_Policy_A` (`29,570` hits, `+353` hits), pero **queda `-527` hits por debajo de `Line3_LightGBM_Enriched` (`30,450` hits, `0.94884` acc, `0.13852` logloss)** y **`-343` hits por debajo de `Line2_CatBoost_Native` (`30,266` hits, `0.94310` acc)** porque prescinde deliberadamente del historial causal del vendedor (`seller_te` / `seller_id`).
   - **Línea 5 (Ensambles simples congelados frente al mejor especialista individual de cada régimen)**:
     - Aunque todos los ensambles de Línea 5 superan ampliamente tanto a `CX0164_Clean_Policy_A` (`+3,511` hits en `unseen_val` con `Q05 = +3,075.59`, `5/5` folds; `+896` hits en `seen_val` con `Q05 = +687.97`, `5/5` folds) como a `T97_Policy_A` (`+6,253` hits en `unseen_val` con `Q05 = +5,635.40`), **la ganancia marginal de `Line5_E5_Selected_Confirm` (`64,256` hits) frente al especialista individual `Line4_NewSeller_Specialist` (`64,246` hits) en `unseen_val` es de sólo `+10` hits con `Q05 = -62.21 < 0`** (`288` fixes / `278` breaks, folds `[+1, -4, +7, +5, +1]`, peor fold `-4`, $\Delta\text{logloss} = -0.00094$).
     - Análogamente, `Line5_E4_Equal_3Tree_Spec` (`64,276` hits) obtiene `+30` hits sobre `Line4_NewSeller_Specialist` pero con **`Q05 = -92.49 < 0`** y peor logloss ($\Delta\text{logloss} = +0.00312$).
     - En `seen_val` (`32,092` filas), `Line5_Seen_Prereg_Blend` (`30,466` hits, `0.94933` acc, `0.13452` logloss) obtiene `+16` hits y $-0.00400$ logloss sobre `Line3_LightGBM_Enriched` (`30,450` hits), pero con **`Q05 = -56.70 < 0`** (`4/5` folds positivos `[+8, -4, +4, +6, +2]`, peor fold `-4`). Por su parte, `Line5_Seen_Dual_LGB_CB` (`30,439` hits, **`-11` hits** vs `Line3_LGB`, `Q05 = -76.01`) y `Line5_Seen_Inner_Convex` (`30,276` hits, **`-174` hits** vs `Line3_LGB`, `0/5` folds positivos, `Q05 = -289.93`) degradan frente a `Line3_LightGBM_Enriched` individual en `seen_val`.
     - Por consiguiente, bajo la regla estricta de no promover por margen, consenso o diversidad cuando el intervalo pareado cruza cero (`Q05 < 0`), **no se reclama superioridad estadística del ensamble sobre `Line4_NewSeller_Specialist` en vendedores nuevos ni sobre `Line3_LightGBM_Enriched` en vendedores vistos**.
4. **Cero generación de portafolio de submissions (`DEVELOPMENT_ONLY`)**:
   - Cumpliendo `CX-0164` ("No usar Kaggle ni generar submissions nuevas en esta etapa", "Estado obligatorio: `DEVELOPMENT_ONLY`"), ningún modelo se promueve a `READY` ni a `submissions/`, y todas las predicciones de test se conservan únicamente como probabilidades de desarrollo y tablas de desacuerdos (`ID` + probabilidades, sin etiquetas inferidas).

---

## 1. Endurecimiento de Gate C (`tests/test_preventive_gates.py`) según `CX-0164`

Se actualizó `validate_gate_c` y `test_gate_c_positive_and_negative_fixtures` en `tests/test_preventive_gates.py` cumpliendo los puntos 1 y E de `CX-0164`:

- **Separación explícita de afirmaciones**:
  - `SCORE_INTERVAL_FIELDS = ("expected_public_score", "public_q05", "public_q95")`: exigen $n \ge \lceil 1/\alpha \rceil = 20$ eventos secuenciales prequential registrados ex-ante (`predicted_before_observing_score == True`) y cobertura empírica recomputada $\ge 0.95$.
  - `THRESHOLD_PROB_FIELDS = ("P(>S4)", "P(>=0.95000)", "prob_beat_s4", "prob_reach_09500")`: además de $n \ge 20$ eventos secuenciales prequential, exigen pronósticos de probabilidad emitidos ex-ante (`prob_forecast` vs `event_outcome`) con verificación de Brier score ($\le 0.20$) y error de calibración empírica ($\le 0.10$). La cobertura de intervalos por sí sola es rechazada para sostener probabilidades de superar un umbral.
- **Fixtures negativos verificados en `test_gate_c_positive_and_negative_fixtures`**:
  1. `neg_uncalibrated_s5`: bloquea `0.95057`, `98.9%`, `66.0%`, `Q05=0.94847` sin calibración.
  2. `neg_5_events_95_bound` (`CX-0164 (i)`): bloquea el uso de $n=5$ eventos para reclamar un límite del `95%`.
  3. `neg_080_coverage_for_095_claim` (`CX-0164 (ii)`): bloquea `0.800` de cobertura empírica cuando se reclama `95%`.
  4. `neg_interval_coverage_for_threshold_prob` (`CX-0164 (iii)`): bloquea presentar cobertura de intervalos (`20/20`) como calibración de `P(>S4)` sin pronósticos probabilísticos ni Brier/calibración.
- **Resultado de ejecución**: `python -m unittest -v tests/test_preventive_gates.py` $\to$ `Ran 8 tests in 1.486s — OK` (`34` aserciones registradas en `results/preventive_gates_repair_20261111/gates_coverage_report.json`).

---

## 2. Protocolo Dual-Regime Congelado (`Unseen-Seller` + `Seen-Seller Causal`) y Auditoría `CX-0164`

Para evaluar simultáneamente vendedores nuevos (`unseen_val`) y vendedores repetidos (`seen_val`) sin filtración temporal ni de identidad (`CX-0164` Sección 3 y `T98`):

1. **Partición compartida congelada (`data/splits/t98_dual_regime_outer_seller_group_splits.csv`, SHA-256 `84643dac738929037d0639cfc177a2af7ce34c4d424ba5537cfeafe85153173f`)**:
   - **5 outer folds (`unseen_val`, `5 x 14,000 = 70,000` filas)**: idéntica asignación `StratifiedGroupKFold(5)` por `seller_id` que `T97` (`0` vendedores compartidos entre `outer-train` y `unseen_val`).
   - **Separación causal dentro de `outer-train` (`56,000` filas por fold)**:
     - Para cada vendedor con $\ge 2$ ítems en `outer-train`, su ítem cronológicamente más reciente (ordenado por `start_time` / `date_created`, desempate por `row`) se separa en **`seen_val`** (`~6,418` filas por fold; **`32,092` filas** sumadas en los 5 folds, `100%` vendedores vistos con $t_{\text{fit}} \le t_{\text{seen\_val}}$).
     - Los ítems anteriores de esos vendedores más los vendedores de ítem único forman **`fit_idx`** (`~49,582` filas por fold; `247,908` filas totales en los 5 folds), dividido a su vez en **4 inner folds agrupados por `seller_id`** (`0` vendedores compartidos entre inner-train e inner-val).
2. **Cuantificación directa del sesgo de `ptxT` in-sample y `seller_te` row-level (`CX-0164` Sección 2)**:
   - **En LightGBM (`Line 3 M0_lgb_cx0164_clean` vs `T97 LightGBM`, mismas 82 features e hiperparámetros Podium)**:
     - Reemplazar `ptxT` in-sample y `seller_te` row-level por `p_text_oof` y `TE` out-of-fold agrupados por `seller_id` (donde `seller_te` en régimen unseen es igual al prior tanto en `fit` como en `unseen_val`, eliminando el shift train/val) eleva los aciertos en `unseen_val` (`70,000` filas) de `58,333` (`0.83333` acc, `0.45167` logloss) a **`60,674` aciertos (`0.86677` acc, `0.31276` logloss)**: **`+2,341` hits** (`4,594` fixes / `2,253` breaks, `5/5` folds positivos `[+502, +393, +414, +633, +399]`, peor fold `+393`, `Q05 = +1,933.86`, $\Delta\text{logloss} = -0.13890$).
   - **En CatBoost (`Line 2 Model_2A_CB_clean_baseline` vs `T97 CatBoost`, mismas 9 features e hiperparámetros Podium)**:
     - Reemplazar `ptxT` in-sample y `cat_te` row-level por `p_text_oof` y `cat_te_oof` agrupados por `seller_id` eleva los aciertos en `unseen_val` de `56,874` (`0.81249` acc, `0.45312` logloss) a **`58,369` aciertos (`0.83384` acc, `0.37728` logloss)**: **`+1,495` hits** (`4,438` fixes / `2,943` breaks, `5/5` folds positivos `[+327, +201, +139, +324, +504]`, peor fold `+139`, `Q05 = +1,041.51`, $\Delta\text{logloss} = -0.07583$).
   - **En el ensamble Podium completo (`CX0164_Clean_Policy_A` vs `T97_Policy_A`)**:
     - Pasa de `58,003` (`0.82861` acc, `0.40147` logloss) a **`60,745` aciertos (`0.86779` acc, `0.31450` logloss)**: **`+2,742` hits** (`4,849` fixes / `2,107` breaks, `5/5` folds positivos `[+591, +443, +527, +709, +472]`, peor fold `+443`, `Q05 = +2,314.46`, $\Delta\text{logloss} = -0.08697$).

---

## 3. Resumen Consolidado de las 5 Líneas (`T98`) y el Rerun Limpio `CX-0164`

### 3.1 Régimen `Unseen-Seller` (`unseen_val`, `70,000` filas, `100%` vendedores nuevos por fold)

| Familia / Modelo | Estado | Hits (`70k`) | Accuracy | Logloss | ROC-AUC | $\Delta$ Hits vs `CX0164_Clean_Policy_A` (`60,745`) | Fixes / Breaks vs Clean A | Folds `+` / `0` / `-` | Peor Fold $\Delta$ | Bootstrap `seller x cat` Mediana (`Q05`, `Q95`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T97_Policy_A` (diagnóstico in-sample `ptxT`) | `DIAGNOSTIC_IN_SAMPLE_STACK_FEATURES` | `58,003` | `0.82861` | `0.40147` | `0.90010` | `-2,742` | `2,107 / 4,849` | `0 / 0 / 5` | `-709` | `-2,715.3` (`-3,330.0`, `-2,314.5`) |
| `T97_Policy_B` (`Dual 50/50` diagnóstico) | `DIAGNOSTIC_IN_SAMPLE_STACK_FEATURES` | `57,967` | `0.82810` | `0.43308` | `0.90488` | `-2,778` | `2,158 / 4,936` | `0 / 0 / 5` | `-702` | `-2,750.1` (`-3,380.2`, `-2,340.1`) |
| `T97_Policy_C` (`Logistic Stack` diagnóstico) | `DIAGNOSTIC_IN_SAMPLE_STACK_FEATURES` | `58,314` | `0.83306` | `0.37780` | `0.91175` | `-2,431` | `2,188 / 4,619` | `0 / 0 / 5` | `-627` | `-2,405.8` (`-3,010.4`, `-2,012.0`) |
| `CX0164_Clean_Text_M0` (Podium Text en `49.6k` fit) | `DEVELOPMENT_ONLY` | `56,327` | `0.80467` | `0.43289` | `0.88133` | `-4,418` | — | `0 / 0 / 5` | `-994` | — |
| `CX0164_Clean_CB_2A` (Podium CB 9-feat OOF limpio) | `DEVELOPMENT_ONLY` | `58,369` | `0.83384` | `0.37728` | `0.91194` | `-2,376` | — | `0 / 0 / 5` | `-534` | — |
| `CX0164_Clean_LGB_M0` (Podium LGB 82-feat OOF limpio) | `DEVELOPMENT_ONLY` | `60,674` | `0.86677` | `0.31276` | `0.93914` | `-71` | — | `0 / 0 / 5` | `-31` | — |
| **`CX0164_Clean_Policy_A` (Baseline limpio corregido)** | `DEVELOPMENT_ONLY` | **`60,745`** | **`0.86779`** | **`0.31450`** | **`0.93871`** | **`0` (Ref.)** | `0 / 0` | — | — | — |
| `CX0164_Clean_Policy_B` (`Dual 50/50` limpio) | `DEVELOPMENT_ONLY` (Degradado) | `60,541` | `0.86487` | `0.32750` | `0.93480` | `-204` | `696 / 900` | `1 / 0 / 4` | `-67` | `-195.4` (`-376.6`, `-54.8`) |
| `CX0164_Clean_Policy_C` (`Logistic Stack` limpio) | `DEVELOPMENT_ONLY` (`Q05 < 0`) | `60,765` | `0.86807` | `0.31129` | `0.93971` | `+20` | `433 / 413` | `3 / 1 / 1` | `-23` | `+19.9` (`-71.2`, `+110.0`) |
| **Línea 1**: `M1_word_char_enriched_saga` | `DEVELOPMENT_ONLY` | `60,585` | `0.86550` | `0.31467` | `0.93941` | `+4,258` vs `Text_M0` | `7,339 / 3,081` vs `Text_M0` | `5 / 0 / 0` | `+663` | `+4,248.8` (`+3,754.7`, `+4,821.2`) vs `Text_M0` |
| **Línea 1**: `M2_subword_char_nb_svm` | `DEVELOPMENT_ONLY` | `60,417` | `0.86310` | `0.32209` | `0.93750` | `+4,090` vs `Text_M0` | `7,232 / 3,142` vs `Text_M0` | `5 / 0 / 0` | `+670` | `+4,078.4` (`+3,604.4`, `+4,619.2`) vs `Text_M0` |
| **Línea 1**: `M3_nested_text_blend` (`M1+M2` inner-frozen) | `DEVELOPMENT_ONLY` | **`60,697`** | **`0.86710`** | **`0.31118`** | **`0.94084`** | **`+4,370` vs `Text_M0`** | `7,430 / 3,060` vs `Text_M0` | `5 / 0 / 0` | `+717` | **`+4,363.0` (`+3,864.8`, `+4,939.5`) vs `Text_M0`** |
| **Línea 2**: `Line2_CatBoost_Native` (CatBoost nativo + texto) | `DEVELOPMENT_ONLY` | **`63,394`** | **`0.90563`** | **`0.23087`** | **`0.96682`** | **`+2,649`** | `4,414 / 1,765` | `5 / 0 / 0` | `+474` | **`+2,642.2` (`+2,299.2`, `+3,028.8`)** |
| **Línea 3**: `Line3_LightGBM_Enriched` (`M1_lgb_enriched`) | `DEVELOPMENT_ONLY` | **`63,996`** | **`0.91423`** | **`0.21547`** | **`0.97123`** | **`+3,251`** | `4,689 / 1,438` | `5 / 0 / 0` | `+592` | **`+3,214.2` (`+2,869.0`, `+3,696.4`)** |
| **Línea 4**: `Line4_NewSeller_Specialist` (`L4_new_seller_blend`) | `DEVELOPMENT_ONLY` | **`64,246`** | **`0.91780`** | **`0.20590`** | **`0.97317`** | **`+3,501`** | `5,108 / 1,607` | `5 / 0 / 0` | `+635` | **`+3,446.8` (`+3,057.0`, `+4,059.4`)** |
| **Línea 5**: `Line5_E5_Selected_Confirm` (inner-frozen confirm) | `DEVELOPMENT_ONLY` | **`64,256`** | **`0.91794`** | **`0.20496`** | **`0.97350`** | **`+3,511`** (`+10` vs L4, `Q05=-62.2`) | `4,989 / 1,478` | `5 / 0 / 0` | `+631` | **`+3,448.1` (`+3,075.6`, `+4,063.3`)** |

---

### 3.2 Régimen `Seen-Seller Causal` (`seen_val`, `32,092` filas: último ítem cronológico de cada vendedor multi-ítem en `outer-train`)

| Familia / Modelo | Estado | Hits (`32,092`) | Accuracy | Logloss | ROC-AUC | $\Delta$ Hits vs `CX0164_Clean_Policy_A` (`29,570`) | Fixes / Breaks vs Clean A | Folds `+` / `0` / `-` | Peor Fold $\Delta$ | Bootstrap `seller x cat` Mediana (`Q05`, `Q95`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `CX0164_Clean_Text_M0` | `DEVELOPMENT_ONLY` | `26,660` | `0.83074` | `0.39709` | `0.89787` | `-2,910` | — | `0 / 0 / 5` | `-611` | — |
| `CX0164_Clean_CB_2A` (causal `seller_id` + OOF `cat_te`) | `DEVELOPMENT_ONLY` | `29,401` | `0.91615` | `0.21497` | `0.96993` | `-169` | — | `0 / 0 / 5` | `-49` | — |
| `CX0164_Clean_LGB_M0` (causal prior-items `seller_te`) | `DEVELOPMENT_ONLY` | `29,575` | `0.92157` | `0.20642` | `0.97224` | `+5` | — | `3 / 0 / 2` | `-9` | — |
| **`CX0164_Clean_Policy_A` (Baseline limpio causal)** | `DEVELOPMENT_ONLY` | **`29,570`** | **`0.92141`** | **`0.20310`** | **`0.97238`** | **`0` (Ref.)** | `0 / 0` | — | — | — |
| `CX0164_Clean_Policy_B` (`Dual 50/50` limpio causal) | `DEVELOPMENT_ONLY` (Degradado) | `29,556` | `0.92098` | `0.20259` | `0.97294` | `-14` | `170 / 184` | `1 / 0 / 4` | `-10` | `-14.6` (`-82.4`, `+54.3`) |
| `CX0164_Clean_Policy_C` (`Logistic Stack` limpio causal) | `DEVELOPMENT_ONLY` (Degradado) | `29,547` | `0.92070` | `0.20405` | `0.97281` | `-23` | `114 / 137` | `1 / 0 / 4` | `-8` | `-22.9` (`-81.8`, `+33.4`) |
| **Línea 1**: `M3_nested_text_blend` | `DEVELOPMENT_ONLY` | **`28,671`** | **`0.89340`** | **`0.26081`** | **`0.95623`** | **`+2,011` vs `Text_M0`** | `2,997 / 986` vs `Text_M0` | `5 / 0 / 0` | `+367` | **`+2,011.2` (`+1,711.4`, `+2,324.4`) vs `Text_M0`** |
| **Línea 4**: `Line4_NewSeller_Specialist` (sin `seller_id`) | `DEVELOPMENT_ONLY` (Inferior a L2/L3 en seen) | `29,923` | `0.93241` | `0.17063` | `0.98119` | `+353` (`-527` vs L3) | — | `5 / 0 / 0` vs Clean A | `+54` | — |
| **Línea 2**: `Line2_CatBoost_Native` (con `seller_id` causal) | `DEVELOPMENT_ONLY` | **`30,266`** | **`0.94310`** | **`0.14609`** | **`0.98643`** | **`+696`** | `1,155 / 459` | `5 / 0 / 0` | `+119` | **`+694.6` (`+494.0`, `+906.0`)** |
| **Línea 3**: `Line3_LightGBM_Enriched` (con `seller_te` causal) | `DEVELOPMENT_ONLY` | **`30,450`** | **`0.94884`** | **`0.13852`** | **`0.98798`** | **`+880`** | `1,171 / 291` | `5 / 0 / 0` | `+160` | **`+878.4` (`+689.1`, `+1,076.6`)** |
| **Línea 5**: `Line5_Seen_Prereg_Blend` (`0.05 L1 + 0.35 L2 + 0.45 L3 + 0.15 L4`) | `DEVELOPMENT_ONLY` | **`30,466`** | **`0.94933`** | **`0.13452`** | **`0.98822`** | **`+896`** (`+16` vs L3, `Q05=-56.7`) | `1,246 / 350` | `5 / 0 / 0` | `+162` | **`+894.1` (`+688.0`, `+1,101.6`)** |

---

## 4. Detalle por Línea (`Líneas 1 a 5`) y Deltas Por Fold (`F1`–`F5`)

### 4.1 Línea 1 — Texto Original (`word+char` SAGA) + Variante `Subword` (`char 2-6` + `char_wb 3-6` NB-SVM)
- **Script**: `scripts/t98_line1_text_subword.py` (`SHA-256: 8e4d9b302e2b8b8ab36380090111bc5f67a92ce51ab1a608a38e0171dbf568e6`)
- **Manifiesto**: `results/t98_line1_text_subword_20261111/manifest.json`
- **Diseño**:
  - `M0_podium_text`: réplica exacta del texto de Podium (`title + tags`, Word 1-2 `40k` + Char_wb 3-5 `60k`, `C=1.5`) entrenado estrictamente en `fit_idx`.
  - `M1_word_char_enriched_saga`: corpus intrínseco enriquecido (`title`, `subtitle`, `warranty`, `tags`, pares `attr_id=value_name`, `listing_type_id`, `buying_mode` y tokens de dominio normalizados), Word(1,3) `80k` + Char_wb(3,5) `80k`, regularización `C` seleccionada en los `4` inner folds.
  - `M2_subword_char_nb_svm`: especialista subword (`char(2,6)` `60k` + `char_wb(3,6)` `60k` = `120k` features) ponderado por el log-count ratio Naive Bayes $\hat{r}_{\text{fit}}$ calculado exclusivamente con etiquetas de `fit_idx` (e inner-train en los inner folds).
  - `M3_nested_text_blend`: combinación convexa/logit de `M1 + M2` seleccionada y congelada en `freeze_line1_outer{0..4}.json` antes de evaluar `unseen_val` y `seen_val`.
- **Deltas por fold de `M3_nested_text_blend` vs `M0_podium_text`**:
  - `unseen_val` (`+4,370` hits, `7,430` fixes / `3,060` breaks, `Q05 = +3,864.8`):
    - Fold 1: `+959` (`1,546` fixes / `587` breaks, $\Delta\text{logloss} = -0.13108$)
    - Fold 2: `+717` (`1,425` fixes / `708` breaks, $\Delta\text{logloss} = -0.10455$)
    - Fold 3: `+895` (`1,428` fixes / `533` breaks, $\Delta\text{logloss} = -0.12352$)
    - Fold 4: `+1,014` (`1,604` fixes / `590` breaks, $\Delta\text{logloss} = -0.13946$)
    - Fold 5: `+785` (`1,427` fixes / `642` breaks, $\Delta\text{logloss} = -0.10995$)
  - `seen_val` (`+2,011` hits, `2,997` fixes / `986` breaks, `Q05 = +1,711.4`):
    - Folds 1–5: `[+367, +435, +400, +396, +413]` (`5/5` positivos, peor fold `+367`, $\Delta\text{logloss} = -0.13628$).
- **Desacuerdos en test (`30,000` filas)**: `results/t98_line1_text_subword_20261111/test_disagreements_vs_s4_probs_only.csv` (`2,728` filas, SHA-256 `3e90a487c8d3616fcd8c8b5601836015d054441cd921aa7bff7307d6ec850736`, columnas: `['ID', 'p_m0_text', 'p_m1_text', 'p_m2_subword', 'p_m3_blend']`).

### 4.2 Línea 2 — CatBoost con Categóricas Nativas y Texto sin Leakage
- **Script**: `scripts/t98_line2_catboost_native.py` (`SHA-256: 05ee56dfe92621d42dd367989b2c5f089b2c5f089d2cf8090706edfab1c4bdba823fded4` — ver `manifest.json` `05ee56dfe92621d42dd367989b2c5f089d2cf8090706edfab1c4bdba823fded4`)
- **Manifiesto**: `results/t98_line2_catboost_native_20261111/manifest.json`
- **Diseño**:
  - `Model_2A_CB_clean_baseline`: las 9 features originales de Podium CatBoost con `p_text_oof` y `cat_te_oof` calculados en 4 inner folds agrupados por `seller_id` (`58,369` hits en `unseen_val`, `+1,495` sobre `T97 CatBoost`).
  - `Model_2B_CB_enriched_unseen_safe` y `Model_2B_CB_enriched_with_seller`: CatBoost (`depth=8`, `1,200` iteraciones) con `cat_features` nativas (`category_id`, `listing_type_id`, `buying_mode`, `ship_mode`, `state_id`, `brand_val`, `item_cond_attr`, `seller_id_masked`), `text_features` nativas (`title`, `tags_attr_text`), precios relativos fold-local, atributos estructurados, garantías, cantidades y `p_text_oof` + `p_attr_oof` estrictamente OOF.
  - En `Line2_Selected_Blend` (pesos congelados en `freeze_line2_outer{0..4}.json` antes de evaluar outer-val):
    - **`unseen_val` (`70,000` filas)**: **`63,394` hits (`0.90563` acc, `0.23087` logloss)**:
      - vs `CX0164_Clean_Policy_A` (`60,745`): **`+2,649` hits** (`4,414` fixes / `1,765` breaks, `5/5` folds `[+504, +474, +479, +629, +563]`, peor fold `+474`, `Q05 = +2,299.17`, $\Delta\text{logloss} = -0.08362$).
      - vs `Model_2A_CB_clean_baseline` (`58,369`): **`+5,025` hits** (`6,960` fixes / `1,935` breaks, `5/5` folds `[+948, +898, +1,111, +1,245, +823]`, peor fold `+823`, `Q05 = +4,314.96`, $\Delta\text{logloss} = -0.14641$).
      - vs `T97_Policy_A` (`58,003`): **`+5,391` hits** (`7,588` fixes / `2,197` breaks, `5/5` folds `[+1,095, +917, +1,006, +1,338, +1,035]`, peor fold `+917`, `Q05 = +4,837.69`).
    - **`seen_val` (`32,092` filas)**: **`30,266` hits (`0.94310` acc, `0.14609` logloss)**:
      - vs `CX0164_Clean_Policy_A` (`29,570`): **`+696` hits** (`1,155` fixes / `459` breaks, `5/5` folds `[+156, +134, +129, +158, +119]`, peor fold `+119`, `Q05 = +493.98`, $\Delta\text{logloss} = -0.05701$).
      - vs `Model_2A_CB_clean_baseline` (`29,401`): **`+865` hits** (`1,367` fixes / `502` breaks, `5/5` folds `[+183, +163, +150, +198, +171]`, peor fold `+150`, `Q05 = +654.40`).
- **Desacuerdos en test (`30,000` filas)**: `results/t98_line2_catboost_native_20261111/test_disagreements_vs_s4_probs_only.csv` (`2,520` filas, SHA-256 `c9237137c5af3a640a588022f7479475c4227f122fb5f7646d226266504dac7f`, columnas: `['ID', 'p_s4_optimal_blend', 'p_s4_cb', 'p_cb_clean_2a', 'p_cb_enriched_unseen_safe', 'p_cb_enriched_with_seller', 'p_cb_line2']`).

### 4.3 Línea 3 — LightGBM Enriquecido con Atributos, Precio Relativo Fold-Local, Garantía, Cantidades y TE OOF
- **Script**: `scripts/t98_line3_lightgbm_enriched.py` (`SHA-256: ce3fd7ca5d98e55393d8a1924763ebc0ebdb6096d5ad452f96f8a40319a44058`)
- **Manifiesto**: `results/t98_line3_lightgbm_enriched_20261111/manifest.json`
- **Diseño**:
  - `M0_lgb_cx0164_clean`: auditoría directa de `CX-0164` sobre las 82 features exactas de Podium LightGBM (`60,674` hits en `unseen_val`, `+2,341` sobre `T97 LightGBM`; `29,575` hits en `seen_val` causal).
  - `M1_lgb_enriched`: `164` features combinando señales multi-vista OOF (`p_text_m0_oof`, `p_text_m1_oof`, `p_subword_oof`, `p_attr_oof`, `SVD(32)` ajustados estrictamente en `fit_idx`), cuantiles/z-scores/ratios de precio por `category_id` y `(category_id, listing_type_id)` fold-local, ratios de cantidad/stock (`sold_quantity`, `available_quantity`, `initial_quantity`, `sell_through`), señales léxicas de garantía y estado, y codificación `TE` régimen-específica (sin `seller_te` en `unseen_val`; con `seller_te` causal de ítems previos cronológicos en `seen_val`), con selección de hiperparámetros en los 4 inner folds (`freeze_line3_outer{0..4}.json`).
  - **Resultados pareados de `M1_lgb_enriched`**:
    - **`unseen_val` (`70,000` filas)**: **`63,996` hits (`0.91423` acc, `0.21547` logloss)**:
      - vs `CX0164_Clean_LGB_M0` (`60,674`): **`+3,322` hits** (`4,735` fixes / `1,413` breaks, `5/5` folds `[+677, +605, +619, +718, +703]`, peor fold `+605`, `Q05 = +2,944.35`, $\Delta\text{logloss} = -0.09730$).
      - vs `CX0164_Clean_Policy_A` (`60,745`): **`+3,251` hits** (`4,689` fixes / `1,438` breaks, `5/5` folds `[+657, +592, +609, +721, +672]`, peor fold `+592`, `Q05 = +2,869.02`, $\Delta\text{logloss} = -0.09903$).
      - vs `T97_Policy_A` (`58,003`): **`+5,993` hits** (`7,898` fixes / `1,905` breaks, `5/5` folds `[+1,248, +1,035, +1,136, +1,430, +1,144]`, peor fold `+1,035`, `Q05 = +5,424.89`).
    - **`seen_val` (`32,092` filas, con `seller_te` causal de ítems previos)**: **`30,450` hits (`0.94884` acc, `0.13852` logloss)**:
      - vs `CX0164_Clean_Policy_A` (`29,570`): **`+880` hits** (`1,171` fixes / `291` breaks, `5/5` folds `[+183, +179, +180, +178, +160]`, peor fold `+160`, `Q05 = +689.05`, $\Delta\text{logloss} = -0.06457$).
      - vs `M0_lgb_cx0164_clean` causal (`29,575`): **`+875` hits** (`1,166` fixes / `291` breaks, `5/5` folds `[+185, +174, +181, +173, +162]`, peor fold `+162`, `Q05 = +682.20`, $\Delta\text{logloss} = -0.06790$).
      - vs `M1_lgb_enriched` sin `seller_te` (`29,783`): **`+667` hits** (`1,259` fixes / `592` breaks, `5/5` folds `[+126, +143, +119, +142, +137]`, peor fold `+119`, `Q05 = +461.64`, $\Delta\text{logloss} = -0.04054$), demostrando el valor del `seller_te` causal cuando el vendedor ya fue observado cronológicamente.
- **Desacuerdos en test (`30,000` filas)**: `results/t98_line3_lightgbm_enriched_20261111/test_disagreements_vs_s4_probs_only.csv` (`3,682` filas, SHA-256 `b5770385ee94748d73d16dbfb184650df1f982a3b2edb1fffe9236cad6ce2071`, columnas: `['ID', 'p_m0_test', 'p_m1_test', 'p_m0_unseen_test', 'p_m1_unseen_test', 'p_m0_seen_causal_test', 'p_m1_seen_causal_test', 'p_s4_lgb', 'p_s4_optimal_blend']`).

### 4.4 Línea 4 — Especialista Intrínseco para Vendedores Nuevos (`0` Variables de Vendedor, `0` Frecuencias Globales)
- **Script**: `scripts/t98_line4_new_seller_specialist.py` (`SHA-256: 6478874a85627c22a34e9a9456a394c84285d26d7ce0e96b07d9f17b9a2a2562`)
- **Manifiesto**: `results/t98_line4_new_seller_specialist_20261111/manifest.json`
- **Diseño**:
  - Elimina por completo cualquier identificador, frecuencia o historial del vendedor (`seller_id_used = False`, `global_frequencies_used = False`), haciendo que la distribución de features sea estructuralmente invariante entre `fit_idx` e ítems de vendedores nunca vistos (`unseen_val`).
  - Construye `221` features puramente intrínsecas al ítem y a la categoría dentro de cada fold: 3 vistas léxicas OOF agrupadas por `seller_id` (texto completo, sólo título, texto estructurado de garantía/atributos/pagos/envíos con `LogisticRegression`, `Ridge`, `ComplementNB` y `TruncatedSVD(64)`), cuantiles empíricos de precio dentro de `category_id` y `(category_id, brand)` calculados sólo en `fit_idx`, Target Encoding jerárquico con shrinkage (`(category, brand) -> category -> prior`) y dos familias de árboles (`L4_lgb_intrinsic` `64,239` hits + `L4_cb_intrinsic` `64,175` hits) combinadas mediante inner-OOF (`L4_new_seller_blend`).
  - **Resultados pareados de `L4_new_seller_blend` en `unseen_val` (`70,000` filas)**: **`64,246` hits (`0.91780` acc, `0.20590` logloss)** — el mejor modelo individual en vendedores nuevos:
    - vs `CX0164_Clean_Policy_A` (`60,745`): **`+3,501` hits** (`5,108` fixes / `1,607` breaks, `5/5` folds `[+674, +635, +650, +776, +766]`, peor fold `+635`, `Q05 = +3,056.96`, $\Delta\text{logloss} = -0.10860$).
    - vs `T97_Policy_A` (`58,003`): **`+6,243` hits** (`8,266` fixes / `2,023` breaks, `5/5` folds `[+1,265, +1,078, +1,177, +1,485, +1,238]`, peor fold `+1,078`, `Q05 = +5,622.99`, $\Delta\text{logloss} = -0.19557$).
    - vs `T97_LightGBM` (`58,333`): **`+5,913` hits** (`7,899` fixes / `1,986` breaks, `5/5` folds `[+1,196, +1,041, +1,074, +1,406, +1,196]`, peor fold `+1,041`, `Q05 = +5,330.83`).
- **Desacuerdos en test (`30,000` filas)**: `results/t98_line4_new_seller_specialist_20261111/test_disagreements_vs_s4_probs_only.csv` (`2,861` filas, SHA-256 `aed34a452647290b14c72dd55640790059f5515fc3df6cd091b5f3e093c801c1`, columnas: `['ID', 'p_l4_new_seller_blend', 'p_lgb_intrinsic', 'p_cb_intrinsic', 'p_text_intrinsic', 'p_s4_podium_blend', 'p_dual_blend_winner', 'abs_prob_diff_vs_s4']`).

### 4.5 Línea 5 — Ensambles Simples Congelados y Rerun Limpio `CX-0164`
- **Script**: `scripts/t98_line5_simple_ensembles.py` (`SHA-256: b9638ca902c62b1c25d1bed5274d56687984dfff041afa33ed390a566d519d68`)
- **Manifiesto**: `results/t98_line5_simple_ensembles_20261111/manifest.json`
- **Diseño**:
  - Combina exclusivamente las 4 líneas especialistas que individualmente superaron al baseline limpio (`Line 1: 60,697`, `Line 2: 63,394`, `Line 3: 63,996`, `Line 4: 64,246`).
  - En cada outer fold $k \in \{1..5\}$, los pesos y el selector `E5_Selected_Confirm` se eligen **únicamente con las predicciones `inner_oof` sobre `fit_idx`** y se congelan en disco antes de evaluar `unseen_val` y `seen_val`.
  - **Resultados en `unseen_val` (`70,000` filas)**:
    - `Line5_E1_Equal_4Way` (`0.25` cada línea): `64,111` hits (`0.91587` acc, `0.22082` logloss).
    - `Line5_E2_Convex_4Way` (pesos convexos inner-OOF): `64,258` hits (`0.91797` acc, `0.20520` logloss).
    - `Line5_E3_Logit_Stack_4Way` (stacking logístico no negativo inner-OOF): `64,260` hits (`0.91800` acc, `0.20487` logloss, `0.97352` AUC).
    - `Line5_E4_Equal_3Tree_Spec` (`1/3 L2 + 1/3 L3 + 1/3 L4`): `64,276` hits (`0.91823` acc, `0.20902` logloss).
    - `Line5_E5_Selected_Confirm` (regla confirm congelada por inner-OOF): **`64,256` hits (`0.91794` acc, `0.20496` logloss)** $\to$ **`+3,511` hits vs `CX0164_Clean_Policy_A`** (`4,989` fixes / `1,478` breaks, `5/5` folds `[+675, +631, +657, +781, +767]`, peor fold `+631`, `Q05 = +3,075.59`), pero sólo `+10` hits con `Q05 = -62.21` frente a `Line4_NewSeller_Specialist` (`64,246`).
  - **Resultados en `seen_val` (`32,092` filas)**:
    - `Line5_Seen_Prereg_Blend` (`0.05 L1 + 0.35 L2 + 0.45 L3 + 0.15 L4`): **`30,466` hits (`0.94933` acc, `0.13452` logloss, `0.98822` AUC)** $\to$ **`+896` hits vs `CX0164_Clean_Policy_A`** (`1,246` fixes / `350` breaks, `5/5` folds `[+191, +175, +184, +184, +162]`, peor fold `+162`, `Q05 = +687.97`), pero sólo `+16` hits con `Q05 = -56.70` frente a `Line3_LightGBM_Enriched` (`30,450`).
- **Desacuerdos en test (`30,000` filas)**: `results/t98_line5_simple_ensembles_20261111/test_disagreements_vs_s4_probs_only.csv` (`774` filas donde `Line5_Regime_Routed` discrepa de `CX0164_Clean_Policy_A`, SHA-256 `e88f7bbeb8b9df849274ab72c77b5e12ab145c125c8937c107a69ad764e3b888`, columnas: `['ID', 'p_CX0164_Clean_Policy_A', 'p_Line5_Unseen_Policy', 'p_Line5_Seen_Policy', 'p_Line5_Regime_Routed']`).

---

## 5. Registro Criptográfico (SHA-256) de Scripts, Manifiestos y Artefactos

| Línea / Componente | Archivo | SHA-256 |
|---|---|---|
| Splits Dual-Regime | `data/splits/t98_dual_regime_outer_seller_group_splits.csv` | `84643dac738929037d0639cfc177a2af7ce34c4d424ba5537cfeafe85153173f` |
| Gate Suite (Gates A–H) | `tests/test_preventive_gates.py` | Verificado (`Ran 8 tests in 1.486s — OK`, `34` fixtures) |
| Línea 1 Script | `scripts/t98_line1_text_subword.py` | `8e4d9b302e2b8b8ab36380090111bc5f67a92ce51ab1a608a38e0171dbf568e6` |
| Línea 1 Unseen OOF | `results/t98_line1_text_subword_20261111/oof_unseen_70k.csv` | `f05ee6e258fd897699c9c8e40138f5902193fe67a3322d046e031e9e751c71af` |
| Línea 1 Seen OOF | `results/t98_line1_text_subword_20261111/oof_seen_all_folds.csv` | `bd1d47fffd9dd760b9d696c7babc613c2a1c849657a0e247ed146b40772d1005` |
| Línea 1 Disagreements | `results/t98_line1_text_subword_20261111/test_disagreements_vs_s4_probs_only.csv` | `3e90a487c8d3616fcd8c8b5601836015d054441cd921aa7bff7307d6ec850736` |
| Línea 2 Script | `scripts/t98_line2_catboost_native.py` | `05ee56dfe92621d42dd367989b2c5f089d2cf8090706edfab1c4bdba823fded4` |
| Línea 2 Unseen OOF | `results/t98_line2_catboost_native_20261111/oof_unseen_70k.csv` | `82ea5e1cc9bdd716163a8086c1ced55d1aad685011575a55530400446693066c` |
| Línea 2 Seen OOF | `results/t98_line2_catboost_native_20261111/oof_seen_all_folds.csv` | `d7c4d23c78454c1f2ba656f9fd4ebaa4c0ea4f29eee0c9480cbec26e353689a1` |
| Línea 2 Disagreements | `results/t98_line2_catboost_native_20261111/test_disagreements_vs_s4_probs_only.csv` | `c9237137c5af3a640a588022f7479475c4227f122fb5f7646d226266504dac7f` |
| Línea 3 Script | `scripts/t98_line3_lightgbm_enriched.py` | `ce3fd7ca5d98e55393d8a1924763ebc0ebdb6096d5ad452f96f8a40319a44058` |
| Línea 3 Unseen OOF | `results/t98_line3_lightgbm_enriched_20261111/oof_unseen_70k.csv` | `72fe07dfea93409db1534b98857967258d206e21082194b8a82da67ae8b7bf26` |
| Línea 3 Seen OOF | `results/t98_line3_lightgbm_enriched_20261111/oof_seen_all_folds.csv` | `268ca2f4dbe159c1ad7d78bba7a878dab6d61b2cce20521879dc9c537810e08f` |
| Línea 3 Disagreements | `results/t98_line3_lightgbm_enriched_20261111/test_disagreements_vs_s4_probs_only.csv` | `b5770385ee94748d73d16dbfb184650df1f982a3b2edb1fffe9236cad6ce2071` |
| Línea 4 Script | `scripts/t98_line4_new_seller_specialist.py` | `6478874a85627c22a34e9a9456a394c84285d26d7ce0e96b07d9f17b9a2a2562` |
| Línea 4 Unseen OOF | `results/t98_line4_new_seller_specialist_20261111/oof_unseen_70k.csv` | `2c0320d576c82751c7d3b1e6ed5d1352082f65660f488b772dbf8c889985855c` |
| Línea 4 Seen OOF | `results/t98_line4_new_seller_specialist_20261111/oof_seen_all_folds.csv` | `a46d52624671cbb3f14509d6d845079755a3344f86b52d7396e7a4f755c04b3c` |
| Línea 4 Disagreements | `results/t98_line4_new_seller_specialist_20261111/test_disagreements_vs_s4_probs_only.csv` | `aed34a452647290b14c72dd55640790059f5515fc3df6cd091b5f3e093c801c1` |
| Línea 5 Script | `scripts/t98_line5_simple_ensembles.py` | `b9638ca902c62b1c25d1bed5274d56687984dfff041afa33ed390a566d519d68` |
| Línea 5 Unseen OOF | `results/t98_line5_simple_ensembles_20261111/oof_unseen_70k.csv` | `86ffc27a061dde3c732bf2063950a5f0c098dcdbb67e374dcd14e17de46f6064` |
| Línea 5 Seen OOF | `results/t98_line5_simple_ensembles_20261111/oof_seen_all_folds.csv` | `82052d1f60fc55052dcccf7bd5d627dde33e85268b59c8652f58560f67d7d5e1` |
| Línea 5 Disagreements | `results/t98_line5_simple_ensembles_20261111/test_disagreements_vs_s4_probs_only.csv` | `e88f7bbeb8b9df849274ab72c77b5e12ab145c125c8937c107a69ad764e3b888` |
