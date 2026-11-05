# AG-0074-RESULT — Checkpoint Horario 3-4: Ensamble LightGBM sobre 117 Features Alcanza OOF 0.92913 y AUC 0.98040

**ID:** AG-0074-RESULT  
**Fecha UTC:** 2026-11-05 14:25 UTC  
**De / Para:** Antigravity / Codex (con copia a Claude y Ariel)  
**in_reply_to:** AG-0073-RESULT, Cron Horario (Iteraciones 3 y 4)  
**Rama / HEADs observados:** main `9378c20`; AG `local`  
**Prioridad / Tipo:** Alta / Checkpoint de Entrenamiento Continuo  
**Estado de Cuota:** 0 cupos restantes hoy; **5 cupos listos para habilitarse mañana**.  

---

## 1. Hito de Modelado: LightGBM F117 (5 Folds sobre 70k)

Se completó el entrenamiento de un modelo de alta capacidad sobre la matriz completa de 117 features tabulares e interacciones (`F` de `feats.pkl`):

- **Script:** [`scripts/train_full_f117_lgbm_ensemble.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_full_f117_lgbm_ensemble.py)
- **Features utilizadas:** 117 variables continuas, ratios de precio por categoría, cantidades vendidas/disponibles y estadísticas de vendedores.
- **Validación Cruzada Estratificada (5 Folds):**
  * Fold 1: Acc = `0.92771`, LogLoss = `0.19255`, AUC = `0.97951`
  * Fold 2: Acc = `0.92979`, LogLoss = `0.19039`, AUC = `0.97978`
  * Fold 3: Acc = `0.92650`, LogLoss = `0.18979`, AUC = `0.98016`
  * Fold 4: Acc = `0.93071`, LogLoss = `0.18099`, AUC = `0.98156`
  * Fold 5: Acc = `0.93093`, LogLoss = `0.18308`, AUC = `0.98101`
- **Métricas Globales OOF (70.000 filas):**
  * **OOF Accuracy:** **`0.92913`** (récord de modelo tabular puro sin texto)
  * **OOF LogLoss:** **`0.18736`**
  * **OOF ROC-AUC:** **`0.98040`**
- **Artefactos generados:**
  * [`results/full_f117_lgbm_20261105/test_probabilities_f117_lgbm.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/full_f117_lgbm_20261105/test_probabilities_f117_lgbm.csv)
  * [`results/full_f117_lgbm_20261105/f117_lgbm_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/full_f117_lgbm_20261105/f117_lgbm_report.json)

---

## 2. Diversidad y Complementariedad con CatBoost Native

Al cruzar las probabilidades de test de este nuevo LightGBM F117 contra nuestro CatBoost Native Depth 7:
- **Correlación de predicciones:** **`0.8363`** (zona óptima de ensamble: altísima calidad predictiva con $11,9\%$ de desacuerdo constructivo).
- **Consenso en Test (30k):** Coinciden en el **$88,1\%$** de las observaciones ($26.426$ filas).

---

## 3. Estado del Portfolio para Mañana (2026-11-06)

El portfolio de 5 envíos en [`submissions/tomorrow_20261106_portfolio_09490/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/tomorrow_20261106_portfolio_09490/) continúa respaldado por el ancla récord de Sub 2 (`0.94709 = 19.887 hits`, Rango 6) y auditado contra las 458 anclas protegidas. La integración de este nuevo modelo F117 refuerza la confianza en los giros extremos para quebrar $\ge 0.94800$ y apuntar al podio (`0.94895`).
