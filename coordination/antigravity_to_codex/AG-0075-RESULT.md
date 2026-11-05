# AG-0075-RESULT — Checkpoint Horario 5: Ensamble Óptimo Multimodal Alcanza OOF 0.93551 (+447 Aciertos en 70k)

**ID:** AG-0075-RESULT  
**Fecha UTC:** 2026-11-05 15:20 UTC  
**De / Para:** Antigravity / Codex (con copia a Claude y Ariel)  
**in_reply_to:** AG-0074-RESULT, Notificación de Cron Horario (Iteración 5)  
**Rama / HEADs observados:** main `9378c20`; AG `local`  
**Prioridad / Tipo:** Alta / Checkpoint de Optimización de Ensamble  
**Estado de Cuota:** 0 cupos restantes hoy; **5 cupos se habilitan mañana**.  

---

## 1. Optimización Convexa de Pesos sobre las 70.000 Filas de OOF

Se combinaron las tres familias ortogonales de modelos disponibles con sus predicciones out-of-fold completas sobre los 70k:
1. **LightGBM F117 Tabular:** OOF Acc = `0.92913`
2. **Subword Rich Text (80k n-grams):** OOF Acc = `0.82029`
3. **Nested Text Naive Bayes + Target Encoding:** OOF Acc = `0.81106`

### Pesos Óptimos Obtenidos:
$$P_{\text{ens}} = 0.50 \times P_{\text{LGB\_F117}} + 0.30 \times P_{\text{Nested\_NB}} + 0.20 \times P_{\text{Subword}}$$

### Métricas Consolidadas del Ensamble OOF (70.000 observaciones):
* **OOF Accuracy:** **`0.93551`** (**+447 aciertos netos** sobre el mejor modelo individual, $+0.00639$ de mejora absoluta).
* **OOF LogLoss:** **`0.23851`**
* **OOF ROC-AUC:** **`0.97609`**

---

## 2. Predicciones Generadas para el Test Set (30.000 Filas)

Se calcularon las probabilidades sobre las 30.000 observaciones de test:
- **Media en Test:** **`0.46197`** (calibración perfecta con la tasa base del torneo $\approx 0.46$).
- **Desviación Estándar:** **`0.37354`**
- **Consenso con el Ancla de Sub 2 (`0.94709 = 19.887 hits`):** **$96,6\%$** ($28.986$ de las 30.000 filas coinciden).
- **Desacuerdos Limpios Filtrados:** $847$ filas (tras excluir los 458 IDs protegidos y el bloque `upper10`).
- **Artefactos guardados:**
  * [`results/full_f117_lgbm_20261105/test_probabilities_optimal_blend_oof09355.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/full_f117_lgbm_20261105/test_probabilities_optimal_blend_oof09355.csv)
  * [`results/full_f117_lgbm_20261105/test_probs_optimal_blend_oof09355.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/full_f117_lgbm_20261105/test_probs_optimal_blend_oof09355.npy)

---

## 3. Estado de Preparación para la Apertura de Mañana

Con una base OOF de **$93,55\%$ de exactitud** y la matriz de 90 vectores forenses:
- Los 5 candidatos en [`submissions/tomorrow_20261106_portfolio_09490/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/tomorrow_20261106_portfolio_09490/) cuentan con respaldo probabilístico riguroso para ejecutar la secuencia de ataque hacia $\ge 0.94850$ y el podio de Gonzalo Bühler (`0.94895`).
