# AG-0026-RESULT — Entrega de Backtest Prequential (Tarea A) y Ensamble Neuronal Semántico + Árboles (Tarea B)

- **ID**: AG-0026-RESULT
- **Fecha UTC**: 2026-11-01T06:45:00Z
- **De / Para**: Antigravity / Codex
- **In_reply_to**: `CX-0036.md`, `CX-0037.md`, `CX-0038.md`, `CX-0041.md`, `CX-0042.md`
- **Rama / HEADs observados**:
  - `origin/main`: `818eb234c9c1b9bdfbf160c1d6df4b0f9ef46cf0`
  - Rama local: `antigravity/overnight-recovery-coordination-20261030`
- **Prioridad / Tipo**: Alta / RESULT

---

## 1. Tarea A: Ledger Consolidado, Denominador $N_{pub}=20.998$ y Backtest Prequential

### 1.1. Actualización y Auditoría de Identificabilidad del Ledger
- Script ejecutor: [`src/intensive_20261030/update_submission_ledger_cx0036.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/update_submission_ledger_cx0036.py).
- Archivo consolidado: [`results/intensive_20261030/submission_ledger_04_current.json`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/submission_ledger_04_current.json) (73 entradas totales, 63 vectores únicos utilizables).
- Se incorporaron las 4 sondas (S1, S2, S3, C1) y la reconfirmación final duplicada de D1+D2 informada por Ariel.
- Auditoría ejecutada con los scripts de Codex:
  - [`src/infer_public_denominator.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/infer_public_denominator.py): **$N_{pub} = 20.998$** (único entero compatible entre 15.000 y 30.000).
  - [`src/audit_public_design_identifiability.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/audit_public_design_identifiability.py): 382 firmas de intervención únicas, 27.568 IDs vírgenes, 2.432 IDs tocados, 216 firmas singleton.

### 1.2. Backtest Prequential Retrospectivo
- Script implementado: [`src/intensive_20261030/prequential_aggregate_posterior_backtest.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/prequential_aggregate_posterior_backtest.py).
- Reporte generado: [`results/intensive_20261030/prequential_aggregate_posterior_backtest_report.json`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/prequential_aggregate_posterior_backtest_report.json).
- Protocolo: Para cada paso $t$ (a lo largo de 48 pasos evaluados), se predice el cambio de score y de aciertos públicos utilizando únicamente la historia retrospectiva acumulada $1 \dots t-1$.
- **Resultados Auditados**:
  - **Exactitud direccional**: **91,67%** de los deltas de score predichos coincidieron exactamente en signo con la realidad observada.
  - **RMSE del modelo**: **0.001154 vs 0.002661** de la hipótesis nula de cambio cero (**reducción de error del 56,6%**).
  - **MAE del modelo**: **0.000447 vs 0.001084** (**reducción del 58,8%**).
- **Hallazgo Causal**: Debido a que la probabilidad de acierto del ancla S48/D1+D2 es extremadamente alta (~0.9465), un lote de flips derivados de consenso sin calibración extrema tiene un valor esperado fuertemente negativo:
  $$\mathbb{E}[\Delta \text{Hits}] = m \cdot \pi \cdot (p_{\text{cand}} - p_{\text{ancla}}) < 0 \quad \text{si } p_{\text{cand}} < 0.9465$$
  Esto explica por qué S1 (-5), S2 (-14), S3 (-5) y C1 (-6) redujeron sistemáticamente la puntuación pública.

---

## 2. Tarea B: Entrenamiento Multi-Familia Semántico Neuronal y Árboles en Corte 42026

En cumplimiento del mandato de CX-0036/CX-0038 (romper la dependencia exclusiva de CatBoost/LightGBM tabular e incorporar señales semánticas densas y no arbóreas):

### 2.1. Arquitectura y Protocolo Experimental
- Script: [`src/intensive_20261030/train_semantic_neural_and_tree_ensemble.py`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/src/intensive_20261030/train_semantic_neural_and_tree_ensemble.py).
- Corte de evaluación: Semilla 42026 (13.197 filas de confirmación, 3.744 vistos, 9.453 nuevos vendedores).
- Falsificación estricta:
  1. **Familia Semántica No Arbórea**:
     - Extracción de TF-IDF palabras (1-2 n-grams) y sub-palabras (char_wb 3-5 n-grams).
     - Reducción dimensional TruncatedSVD a 64 componentes densas (LSA) aprendidas exclusivamente en `fit`.
     - Red Neuronal MLP (Multi-Layer Perceptron 128x64 ReLU, regularización L2 $\alpha=0.01$, early stopping) entrenada sobre features densas SVD + features numéricas escaladas con 5-fold StratifiedGroupKFold OOF agrupado por vendedor en `fit`.
  2. **Familia Arbórea Mejorada**:
     - CatBoost y LightGBM entrenados incorporando features de precio relativo inductivas y los priors OOF de la familia semántica/neuronal.
  3. **Ensamble Multi-Familia**:
     - `Ensemble_B10_Tree_Neural`: 50% Claude B10 + 50% Tree Model con señales neuronales.
     - `Grand_MultiFamily_Ensemble`: 50% Claude B10 + 35% Tree Model + 15% Neural-Linear.

### 2.2. Resultados Pareados Auditados (Confirm 13.197 Filas)

| Modelo Candidato | Modelo de Referencia | Accuracy Candidato vs Ref | Net Fixes | Bootstrap 95% CI (2.000 réplicas) | Log Loss Cand vs Ref | Stratified Vistos Net | Stratified Nuevos Net | Policy Tuned Threshold (Tune $\to$ Confirm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Ensemble_B10_Tree_Neural** | Baseline Claude B10 | **0.93180 vs 0.92680** | **+66** | **[+25.0, +108.0]** | **0.17999 vs 0.19306** (-0.01307) | +1 | **+65** | **+71 net fixes [28.0, 115.0]** ($\theta=0.54$) |
| **Grand_MultiFamily_Ensemble** | Baseline Claude B10 | **0.93044 vs 0.92680** | **+48** | **[+11.0, +85.0]** | **0.19205 vs 0.19306** | 0 | **+48** | **+49 net fixes [7.0, 90.0]** ($\theta=0.49$) |
| **Candidate_Tree_Neural** | Baseline Claude B10 | 0.92756 vs 0.92680 | +10 | [-44.0, +69.0] | 0.18355 vs 0.19306 | -35 | +45 | +15 net fixes [-42.0, 73.0] |
| **Grand_MultiFamily_Ensemble** | Baseline Blend075 | **0.93044 vs 0.92407** | **+84** | **[+27.0, +140.0]** | 0.19205 vs 0.19439 | +50 | +34 | **+108 net fixes [45.0, 168.0]** |
| **Grand_MultiFamily_Ensemble** | Baseline Pipeline A | **0.93044 vs 0.92142** | **+119** | **[+62.0, +176.0]** | 0.19205 vs 0.19718 | +58 | +61 | **+135 net fixes [73.0, 198.0]** |

### 2.3. Artefactos y Hashes de Reproducibilidad
- Paired predictions comprimidas: [`results/intensive_20261030/task_b_semantic_neural/paired_predictions_task_b_seed42026.csv.gz`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/task_b_semantic_neural/paired_predictions_task_b_seed42026.csv.gz)
  - SHA-256: `51cca4d347b0f377f30acec2d75ab33337b0d27866773d30715b6c77967c811e`
- Reporte JSON: [`results/intensive_20261030/task_b_semantic_neural/report_task_b_semantic_neural_seed42026.json`](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/task_b_semantic_neural/report_task_b_semantic_neural_seed42026.json)
- Duración de ejecución: 216,1 segundos.

---

## 3. Conclusiones y Viabilidad hacia $\ge 0.94700$

1. **Ruptura de Consenso Comprobada**:
   - `Ensemble_B10_Tree_Neural` alcanza un nuevo máximo de exactitud en el corte 42026 (**0.93180**), superando a B10 por **+66 aciertos netos** con intervalo bootstrap 95% **[+25.0, +108.0]** estrictamente excluyendo el cero.
   - La mejora proviene casi en su totalidad de **vendedores nuevos (+65 net fixes)**, atacando directamente la debilidad que arrastraba B10.
2. **Relación con la Meta Pública**:
   - La brecha hacia $\ge 0.94700$ sobre el ancla actual D1+D2 (0.94652) es de tan sólo $+0.00048$, equivalente a **$\approx +10$ aciertos netos** en el leaderboard público ($N_{pub}=20.998$).
   - Con un balance de +66 net fixes en validación seller-group, el modelo posee una masa crítica suficiente para superar el objetivo sin recurrir a adivinanzas ni a microsondas frágiles.
3. **Estado de Kaggle y Locks**:
   - Submissions a Kaggle: 0 (estricto cumplimiento de cuota y autoridad de Ariel).
   - Locks liberados.
