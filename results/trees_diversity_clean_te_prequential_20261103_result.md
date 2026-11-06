# Diversidad Estructural de Árboles (CatBoost + LightGBM): TE Limpio y Evaluación Prequential en 3 Cortes

**Fecha:** 2026-11-03 (Ejecución completada 2026-11-04 02:09 UTC)  
**Script:** [`scripts/audit_trees_clean_te_prequential_20261103.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_trees_clean_te_prequential_20261103.py) (SHA-256: `3a57a6636d4151be086f555c80d838edfc630a0acdfaec376374733fe089f75e`)  
**Reporte JSON:** [`results/multimodel_trees_lgbm_catboost/report_trees_diversity_clean_te_prequential.json`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/multimodel_trees_lgbm_catboost/report_trees_diversity_clean_te_prequential.json)  
**Respuesta a:** CX-0093, CX-0094, CX-0095.

---

## 1. Auditoría Empírica de Diversidad de Semillas en LightGBM (CX-0093 A)

Se auditó si `random_state` produce diversidad efectiva en LightGBM bajo `colsample_bytree: 0.8` (`bagging_freq=0`):

| Split / Semillas | Iteraciones Óptimas | Correlación Pearson (mín/máx) | Máxima Dif. Absoluta de Prob. | Mediana Dif. Absoluta | Desacuerdos Binarios @ 0,50 |
|---|---|---|---|---|---|
| **Corte 20261141** | [456, 401, 500] | 0,99840 – 0,99845 | 0,4673 | 0,00108 – 0,00155 | 80 a 96 (0,64% – 0,77%) |
| **Corte 20261143** | [440, 408, 402] | 0,99870 – 0,99878 | 0,3768 | 0,00108 – 0,00110 | 69 a 89 (0,56% – 0,72%) |
| **Corte 20261144** | [369, 336, 333] | 0,99866 – 0,99876 | 0,5025 | 0,00133 – 0,00136 | 76 a 100 (0,62% – 0,82%) |

**Comparador CatBoost Bag3 (3 semillas)**: Correlación Pearson 0,9977 – 0,9983, máx dif 0,29 – 0,43, desacuerdos 89 a 119 por corte.  
**Conclusión de Diversidad**: Las tres semillas de LightGBM **no son copias idénticas**; presentan diferencias máximas de probabilidad de hasta 0,5025 y entre 69 y 100 desacuerdos binarios por corte debido al submuestreo de variables por árbol (`colsample_bytree: 0.8`). Su nivel de diversidad es comparable al del propio CatBoost Bag3.

---

## 2. Impacto del Target Encoding Limpio (Prior Fold-Específico) vs AG-0047 en Semilla 20261144

Se compararon las predicciones de confirm (12.264 filas) con el prior corregido `y[fit[tr]].mean()` vs las de AG-0047 (prior global contaminado):

- **CatBoost Bag3**: Cambio medio de probabilidad de **0,00579** (máx 0,1728).
  - Accuracy confirm pasa de 0,955887 (11.723 aciertos) a **0,955153 (11.714 aciertos)**: neto **-9 aciertos** en el baseline.
- **Blend 80/20**: Cambio medio de probabilidad de **0,00482** (máx 0,1449).
  - Accuracy confirm a umbral propio tune (0,475): **0,956050 (11.725 aciertos)**, delta = **+11 aciertos** vs CB, CI 95% `[-6.0, +30.0]`.
  - Accuracy confirm a umbral común 0,500: **0,956295 (11.728 aciertos)**, delta = **+14 aciertos** vs CB, CI 95% `[-1.0, +29.0]`.
  - Accuracy confirm a umbral transportado de 20261143: **0,956866 (11.735 aciertos)**, delta = **+25 aciertos** vs CB, CI 95% `[0.0, +51.0]`.
  - LogLoss: 0,12034 $\to$ **0,11892** (mejora de **-0,00142**).

---

## 3. Desempeño por Corte en los 3 Cortes Históricos

### Corte 1: 20261141 (Confirm: 12.448 filas)
- **A umbral 0,500**: CatBoost = 11.912 | Blend 80/20 = 11.898 | **Delta = -14 aciertos**, CI 95%: `[-30.0, 2.0]`
- **A umbral propio tune (CB=0,500, B80=0,545)**: CatBoost = 11.912 | Blend 80/20 = 11.904 | **Delta = -8 aciertos**, CI 95%: `[-26.0, 9.0]`
- **LogLoss**: CB = 0,11845 | B80 = **0,11768** (-0,00077).

### Corte 2: 20261143 (Confirm: 12.360 filas)
- **A umbral 0,500**: CatBoost = 11.703 | Blend 80/20 = 11.726 | **Delta = +23 aciertos**, CI 95%: `[7.0, 39.0]`
- **A umbral propio tune (CB=0,425, B80=0,525)**: CatBoost = 11.705 | Blend 80/20 = 11.718 | **Delta = +13 aciertos**, CI 95%: `[-8.0, 35.0]`
- **A umbral transportado de Corte 1 (CB=0,500, B80=0,545)**: CatBoost = 11.703 | Blend 80/20 = 11.721 | **Delta = +18 aciertos**, CI 95%: `[2.0, 35.0]`
- **LogLoss**: CB = 0,13531 | B80 = **0,13411** (-0,00120).

### Corte 3: 20261144 (Confirm: 12.264 filas)
- **A umbral 0,500**: CatBoost = 11.714 | Blend 80/20 = 11.728 | **Delta = +14 aciertos**, CI 95%: `[-1.0, 29.0]`
- **A umbral propio tune (CB=0,500, B80=0,475)**: CatBoost = 11.714 | Blend 80/20 = 11.725 | **Delta = +11 aciertos**, CI 95%: `[-6.0, 30.0]`
- **A umbral transportado de Corte 2 (CB=0,425, B80=0,525)**: CatBoost = 11.710 | Blend 80/20 = 11.735 | **Delta = +25 aciertos**, CI 95%: `[0.0, 51.0]`
- **LogLoss**: CB = 0,12034 | B80 = **0,11892** (-0,00142).

---

## 4. Análisis Prequential y Deduplicado Global (29.669 ítems únicos)

Al consolidar los 3 cortes de manera prequential y deduplicar por `item_id`:

| Regla de Evaluación | Aciertos CatBoost Bag3 | Aciertos Blend 80/20 | Delta Neto | Fixes / Breaks | Bootstrap 95% CI Vendedor (3.000 réplicas) | LogLoss CB $\to$ Blend |
|---|---|---|---|---|---|---|
| **Umbral Fijo 0,500** | 28.229 (0,95146) | 28.241 (0,95187) | **+12** | 76 / 64 | **`[-13.0, +37.0]`** | 0,12757 $\to$ **0,12638** (-0,00119) |
| **Umbral Transportado Prequential** | 28.222 (0,95123) | 28.250 (0,95217) | **+28** | 124 / 96 | **`[-1.0, +59.0]`** | — |

---

## 5. Dictamen Científico y Decisión de Kaggle

1. **Efecto en LogLoss**: El blend árbol oblivious + árbol leaf-wise mejora el logloss de forma monotónica y consistente en los 3 cortes históricos (reducción promedio de -0,00119).
2. **Efecto en Accuracy**:
   - En Corte 2 (+23 @ 0,50) y Corte 3 (+14 @ 0,50; +25 transportado), el blend es positivo.
   - En Corte 1 (20261141), el blend pierde -14 aciertos a umbral 0,50 y -8 a umbral tune.
   - En la evaluación deduplicada global de 29.669 ítems, el intervalo bootstrap del 95% por vendedor cruza el cero (`[-13.0, +37.0]` a 0,50 y `[-1.0, +59.0]` transportado).
3. **Decisión**: Conforme a los criterios de CX-0093 y CX-0095, la mejora de logloss justifica mantener el ensamble como componente interno, pero la variabilidad en accuracy y la falta de significancia estricta (CI tocando o cruzando cero) **no autorizan una submission pública**.
4. **Estado de Kaggle**: **0 envíos realizados. 50/50 cuotas preservadas al 100%.**
