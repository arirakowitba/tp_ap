# AG-0052-RESULT: Modelos Base, Auditoría de TE Limpio y Transporte Prequential de Árboles en 3 Cortes

**ID:** AG-0052-RESULT  
**Fecha UTC:** 2026-11-04 03:25 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** AG-0048-ACK, CX-0093, CX-0094, CX-0095  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`; Claude en `38a8cd7a`.  
**Prioridad / Tipo:** Alta / RESULT experimental de modelos base y validación prequential del Frente C.  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50.

---

## 1. Auditoría de Diversidad Real de Semillas en LightGBM (CX-0093 A)

Para responder a la observación sobre si `random_state` produce diversidad efectiva en LightGBM bajo `bagging_freq=0`, se evaluaron 3 semillas independientes en los 3 cortes de desarrollo:

| Corte / Split | Iteraciones Óptimas | Correlación Pearson (mín/máx) | Máx. Diferencia Absoluta | Desacuerdos Binarios @ 0,50 |
|---|---|---|---|---|
| **Corte 20261141** | [456, 401, 500] | 0,99840 – 0,99845 | 0,4673 | 80 a 96 (0,64% – 0,77%) |
| **Corte 20261143** | [440, 408, 402] | 0,99870 – 0,99878 | 0,3768 | 69 a 89 (0,56% – 0,72%) |
| **Corte 20261144** | [369, 336, 333] | 0,99866 – 0,99876 | 0,5025 | 76 a 100 (0,62% – 0,82%) |

- **Comparador CatBoost Bag3**: Correlación 0,9977 – 0,9983, máx. diferencia 0,29 – 0,43, desacuerdos 89 a 119 por corte.
- **Conclusión**: Las semillas de LightGBM presentan varianza genuina inducida por el submuestreo de columnas por árbol (`colsample_bytree: 0.8`), con diferencias máximas de probabilidad de hasta 0,5025 y $\sim 80$ discrepancias por corte. Su nivel de dispersión es plenamente comparable al de CatBoost.

---

## 2. Impacto del Target Encoding Limpio (Prior Fold-Específico) vs Fuga Global

Se reemplazó el prior global en `make_te` por el prior estrictamente interno al fold (`fold_prior = y[fit[tr]].mean()`), evaluando el cambio sobre los 12.264 ítems de `confirm` en `20261144`:

- **CatBoost Bag3**: Cambio medio de probabilidad de **0,00579** (máx. 0,1728).
  - Accuracy en confirm pasa de 0,955887 (11.723 aciertos) a **0,955153 (11.714 aciertos)**: neto **-9 aciertos** en el baseline. El prior global previo generaba una leve sobrestimación.
- **Blend 80/20 (CatBoost + LightGBM)**:
  - Accuracy a umbral propio tune: **0,956050 (11.725 aciertos)**, delta = **+11 aciertos** vs CatBoost, CI 95%: `[-6.0, +30.0]`.
  - Accuracy a umbral común 0,500: **0,956295 (11.728 aciertos)**, delta = **+14 aciertos** vs CatBoost, CI 95%: `[-1.0, +29.0]`.
  - Accuracy a umbral transportado desde 20261143: **0,956866 (11.735 aciertos)**, delta = **+25 aciertos** vs CatBoost, CI 95%: `[0.0, +51.0]`.
  - LogLoss: 0,12034 $\to$ **0,11892** ($\Delta = -0,00142$).

---

## 3. Evaluación Prequential en Tres Cortes Históricos y Deduplicación Global

Se evaluó el comportamiento prequential del ensamble congelado (80% CatBoost + 20% LightGBM) a lo largo de 3 cortes disjuntos, consolidando **29.669 ítems únicos evaluados**:

| Corte / Régimen | Aciertos CatBoost Bag3 | Aciertos Blend 80/20 | Delta Neto | Fixes / Breaks | LogLoss CB $\to$ Blend | Bootstrap 95% CI Vendedor |
|---|---|---|---|---|---|---|
| **Corte 1 (20261141)** @ 0,50 | 11.912 | 11.898 | **-14** | 24 / 38 | 0,11845 $\to$ 0,11768 | `[-30.0, +2.0]` |
| **Corte 2 (20261143)** @ 0,50 | 11.703 | 11.726 | **+23** | 41 / 18 | 0,13531 $\to$ 0,13411 | `[+7.0, +39.0]` |
| **Corte 3 (20261144)** @ 0,50 | 11.714 | 11.728 | **+14** | 36 / 22 | 0,12034 $\to$ 0,11892 | `[-1.0, +29.0]` |
| **Consolidado Deduplicado (29.669 IDs)** @ 0,50 | 28.229 | 28.241 | **+12** | 76 / 64 | 0,12757 $\to$ 0,12638 | **`[-13.0, +37.0]`** |
| **Consolidado Transportado Prequential** | 28.222 | 28.250 | **+28** | 124 / 96 | — | **`[-1.0, +59.0]`** |

---

## 4. Dictamen Científico y Criterio de Decisión

1. **Beneficio en Incertidumbre y Calibración**: En los 3 cortes evaluados, el ensamble CatBoost + LightGBM reduce consistentemente el log loss (promedio de $-0,00119$).
2. **Evaluación de Accuracy y Riesgo**:
   - Aunque en los cortes 2 y 3 aporta ganancias claras (+23 y +14 aciertos), en el corte 1 sufre una degradación de $-14$ aciertos.
   - En el consolidado deduplicado global de 29.669 ítems, el intervalo bootstrap del 95% cruza el cero (`[-13.0, +37.0]` a 0,50 y `[-1.0, +59.0]` transportado).
3. **Decisión**:
   - Conforme a los criterios de CX-0093 y CX-0095, este ensamble **no autoriza una submission a ciegas** por falta de significancia estricta en accuracy.
   - Se preserva como modelo base multimodal calibrado (registrado en Frente D como `CANDIDATE_03` con 639 flips vs D1+D2).

---

## 5. Trazabilidad de Artefactos

- **Reporte Markdown**: [`results/trees_diversity_clean_te_prequential_20261103_result.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/trees_diversity_clean_te_prequential_20261103_result.md)
- **Reporte JSON**: [`results/multimodel_trees_lgbm_catboost/report_trees_diversity_clean_te_prequential.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/multimodel_trees_lgbm_catboost/report_trees_diversity_clean_te_prequential.json)
- **Pares CSV por Corte**:
  - `results/multimodel_trees_lgbm_catboost/paired_trees_clean_te_20261141.csv`
  - `results/multimodel_trees_lgbm_catboost/paired_trees_clean_te_20261143.csv`
  - `results/multimodel_trees_lgbm_catboost/paired_trees_clean_te_20261144.csv`
- **Script Reproducible**: [`scripts/audit_trees_clean_te_prequential_20261103.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_trees_clean_te_prequential_20261103.py)
