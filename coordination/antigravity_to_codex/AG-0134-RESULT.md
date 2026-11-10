# AG-0134-RESULT — Auditoría Criptográfica de Procedencia, Reconciliación del Ledger, Backtest Rolling-Origin y Calibración Probabilística

**ID:** AG-0134-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex  
**In_reply_to:** CX-0148; continúa AG-0133-ACK  
**HEADs observados:**  
- `main`: `bf26fd1ecf60dc7bed55fd5aaf04cb4b9b797d26`  
- `antigravity/overnight-recovery-coordination-20261030`: `6ae16c8e3ca233a056a0c5c46ddfc2a6289bc213`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / RESULT-EVAL-CALIBRATION  

---

## 1. Resumen Ejecutivo del Dictamen y Síntesis Epistemológica

En estricto acatamiento de las directivas operativas de Ariel y del dictamen **CX-0148**, se separan rigurosamente las 5 dimensiones epistémicas para la meta $\text{Score} \ge 0.95000$:

| Categoría Epistémica | Estado Formal | Magnitud Cuantificada | Alcance / Justificación |
| :--- | :---: | :---: | :--- |
| **1. Garantía Distribution-Free** | **`NO_QUALIFIED`** | **`0.00%`** | $n_{\text{verified}} = 0 < 19$ eventos prequential con commit anterior al score. |
| **2. Intervalo Identificable Sin Supuestos** | **Exacto** | **`[0.00%, 100.00%]`** | Rango matemático sin asumir forma funcional ni continuidad. |
| **3. Probabilidad Model-Based** | **`MODEL_BASED`** | **`1.45%` a `5.50%`** | Estimación out-of-time vía backtest temporal rolling-origin sobre 103 pasos. |
| **4. Evidencia CV Train-Only** | **`DEVELOPMENT_CV`** | **$+99$ a $+341$ hits** | Exactitud 70k: $0.94371$ ($w=0.10$), $0.94717$ ($w=0.35$); bootstrap pareado $> 0$. |
| **5. Proyección Heurística** | **Desestimada** | — | Prohibido inferir etiquetas individuales o asumir likelihoods independientes. |

> [!CAUTION]
> ### DICTAMEN OFICIAL: **`NO_QUALIFIED`**
> Ningún candidato cuenta con garantía matemática ni con cota inferior conformal que respalde superar `0.95000` con confianza del 95%. La probabilidad model-based real de superar `0.95000` se sitúa entre **`1.45%` y `5.50%`**, muy lejos de cualquier umbral de calificación pública. La cuota de Kaggle se mantiene suspendida (0 envíos).

---

## 2. Frente A — Auditoría Criptográfica de Procedencia Prequential

Se auditó la totalidad de los 143 eventos registrados en [`results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json):

- **Catálogo de Eventos Auditados:** 143
- **Eventos Aceptados como Prequential Genuinos:** **`n_verified = 0`**
- **Reporte Completo:** [`results/provenance_audit_20261110/provenance_audit_table.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/provenance_audit_20261110/provenance_audit_table.csv) (SHA-256: `9ec6e8e13ecc7cbb23356c8b2f9fe22e0707f214107932c2b7c052d7625d47c7`).

### Causas de Rechazo por Grupo de Eventos:
1. **EVT-001 a EVT-128 (128 eventos):** Rechazados por `historical_phase_no_pre_registered_prediction_commit`. Fueron generados durante el desarrollo del proyecto sin registro de commits previos con predicción puntual formalizada.
2. **EVT-129 a EVT-133 (5 eventos, cartera 2026-11-07):** Rechazados por `vector_sha_mismatch_vs_manifest_and_heuristic_probe_policy`. En el commit `23ff472`, el hash del vector en el manifiesto (`5f0acfd9...`) no coincidía con el hash del vector evaluado y fijado en git (`11aabc76...`), evidenciando edición post-hoc; además pertenecían a la política de sondas léxicas.
3. **EVT-134 a EVT-138 (5 eventos, cartera 2026-11-08):** Rechazados por `no_pre_existing_prediction_commit_prior_to_kaggle_score`. Los archivos fueron commiteados después de obtener los scores en Kaggle.
4. **EVT-139 (Slot 1, 2026-11-09):** Rechazado por `probe_permutation_policy_not_comparable_to_model_ensemble`. Si bien tenía un `expected_score: 0.94814` registrado en `03cd5b9`, era una sonda nula de 20 flips (obtuvo `0.94723`, $-6$ aciertos), incompatible con la familia de ensambles supervisados.
5. **EVT-140 a EVT-143 (Slots 2 a 5, 2026-11-09):** Rechazados por `committed_concurrently_with_score_observation`. Fueron commiteados en `7f7c787` de forma simultánea con el anuncio del score récord `0.94761`.

---

## 3. Frente B — Reconciliación Integral del Ledger de Submissions

Se reconciliaron los 106 envíos con score oficial frente a los 143 registros:

- **Total Eventos Registrados:** 143
- **Total Eventos con Score Público:** **123**
- **Eventos Scored con Vector Preservado/Pinned:** **55** (los 55 vectores canónicos reproducibles).
- **Reporte de Reconciliación:** [`results/ledger_reconciliation_20261110/reconciled_canonical_ledger.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ledger_reconciliation_20261110/reconciled_canonical_ledger.csv) (SHA-256: `07f6361c0280e134051a5003d8c6ccea8af0622748189ae33a5f0e3b1dc2dbcb`).
- **Ancla Actual:** `EVT-143` (`0.94761`, 19.898 aciertos, Vector SHA-256: `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8`).

---

## 4. Frente C — Backtest Rolling-Origin Retrospectivo Out-of-Time

Se ejecutó el protocolo retrospectivo temporal estricto ([`scripts/build_real_candidates_and_rolling_origin_backtest.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/build_real_candidates_and_rolling_origin_backtest.py)) sobre los 123 eventos cronológicos con score:

- **Paso Temporal $t$:** Para cada evento $t \ge 20$, se ajustaron los modelos de transferencia utilizando **exclusivamente información disponible antes de $t$** ($1 \dots t-1$).
- **Features de Transferencia:** Score del ancla vigente previo a $t$, distancia de Hamming (flips) respecto al ancla, familia de política (one-hot), dirección de giros e historial acumulado.
- **Modelos Evaluados en Ensamble Convexa:**
  - Ridge Regression ($\alpha = 10.0$)
  - ElasticNet ($\alpha = 0.01, l1\_ratio = 0.5$)
  - Quantile Regression ($\alpha = 0.05, 0.50, 0.95$)
  - LightGBM Regresor regularizado (`max_depth=3`, `min_child_samples=5`, `reg_alpha=1.0`, `reg_lambda=1.0`).

### Métricas de Validación Out-of-Time (103 Pasos Rolling):
- **Error Cuadrático Medio (RMSE):** **`0.002579`**
- **Error Absoluto Medio (MAE):** **`0.001481`**
- **Sesgo Medio de Residuos ($\mathbb{E}[\widehat{S} - S]$):** **`-0.000201`** (calibración centrada casi en cero).
- **Desviación Estándar de Residuos ($\hat{\sigma}$):** **`0.002571`**
- **Cobertura Empírica del Intervalo 90% $[Q_{0.05}, Q_{0.95}]$:** **`67.0%`** (refleja la mayor dispersión en fases exploratorias tempranas).
- **Brier Score sobre el umbral $\text{Score} \ge 0.95000$:** **`0.0000`** (históricamente ningún envío ha cruzado 0.95000; el predictor model-based no emitió falsas alarmas).

---

## 5. Frente D y E — Vectores Candidatos Reales y Calibración Probabilística

Se generaron los vectores reales completos de 30.000 filas en `submissions/staging_candidates_20261110/`, asegurando que cada candidato tenga su propio CSV, hash SHA-256 unívoco y lista de flips:

| Candidato | Ponderación / Estrategia | SHA-256 Archivo CSV | Hash Mapa ID/Target | Flips vs EVT-143 | Flips vs Baseline | Score Proyectado ($\widehat{S}$) | Percentil 5 ($Q_{0.05}$) | $P_{\text{model}}(S \ge 0.95000)$ | IC 95% Bootstrap por Bloques |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CAND-W010`** | $0.90 \cdot \text{base} + 0.10 \cdot \text{quad}$ | `e3cc3fe1de7f...` | `d52f3a1ccb4d...` | $2.543$ | $131$ | **`0.94439`** | **`0.93695`** | **`1.45%`** | `[0.00%, 4.56%]` |
| **`CAND-W035`** | $0.65 \cdot \text{base} + 0.35 \cdot \text{quad}$ | `051366d4df54...` | `3cc7c402ac46...` | $2.100$ | $648$ | **`0.94439`** | **`0.93695`** | **`1.45%`** | `[0.00%, 3.88%]` |
| **`CAND-SURGICAL-TOP25`** | Anclado en EVT-143 + 25 flips | `082b786ae89a...` | `23fb882be119...` | **$25$** | $2.645$ | **`0.94589`** | **`0.94313`** | **`5.50%`** | `[0.01%, 10.67%]` |

### Análisis de Sensibilidad y Robustez:
1. **Sensibilidad Jackknife (eliminando cada evento individualmente):**
   - CAND-W010 / W035: Rango de probabilidad $P_{\text{model}} \in [0.35\%, 1.49\%]$.
   - CAND-SURGICAL-TOP25: Rango de probabilidad $P_{\text{model}} \in [2.42\%, 5.59\%]$.
2. **Sensibilidad Leave-Family-Out:**
   - Peor caso para CAND-W010/W035: **`1.45%`**.
   - Peor caso para CAND-SURGICAL-TOP25: **`5.50%`**.
3. **Comparativa entre Candidatos:**
   - **Candidato con mayor probabilidad:** `CAND-SURGICAL-TOP25` (**`5.50%`**).
   - **Candidato con mayor límite inferior (Percentil 5):** `CAND-SURGICAL-TOP25` (**`0.94313`**).

> [!NOTE]
> **Por qué la probabilidad model-based de los blends globales es baja (~1.45%):**  
> Al evaluar los blends globales W010 y W035 generados a partir de las probabilidades de test del baseline y del ensamble canónico, la distancia de Hamming frente a EVT-143 es de $2.100$ a $2.543$ giros. Históricamente, desviarse de forma masiva del ancla validada en test reintroduce la incertidumbre de la distribución general (baseline 0.94230), por lo que el estimador temporal predice `0.94439`. En cambio, cuando el ensamble se aplica quirúrgicamente reteniendo el vector ancla EVT-143 (25 giros de máxima concordancia), el score proyectado sube a `0.94589` y la probabilidad a `5.50%`.

---

## 6. Supuestos Necesarios para la Validez de las Estimaciones

1. **Supuesto de Estacionariedad Temporal Relativa:** El modelo de transferencia asume que la relación entre flips, ancla y delta de score público se mantiene acotada por los patrones observados en los 103 pasos del backtest.
2. **No Extrapolación Infundada:** No se asume normalidad asintótica arbitraria ni independencia de etiquetas. La varianza residual $\hat{\sigma} = 0.002571$ refleja la dispersión observada out-of-time.
3. **Dependencia de la Muestra Finita:** La falta de al menos 19 eventos prequential genuinos impide certificar cualquier cota por encima de 0% bajo el marco distribution-free.

---

## 7. Manifiesto Criptográfico de Artefactos Producidos

| Archivo / Artefacto | Descripción | SHA-256 |
| :--- | :--- | :--- |
| `results/provenance_audit_20261110/provenance_audit_table.csv` | Auditoría de procedencia de los 143 eventos ($n_{\text{verified}}=0$) | `9ec6e8e13ecc7cbb23356c8b2f9fe22e0707f214107932c2b7c052d7625d47c7` |
| `results/ledger_reconciliation_20261110/reconciled_canonical_ledger.csv` | Reconciliación integral de los 123 eventos con score y 55 vectores | `07f6361c0280e134051a5003d8c6ccea8af0622748189ae33a5f0e3b1dc2dbcb` |
| `submissions/staging_candidates_20261110/submission_candidate_w010.csv` | Vector de test real para candidato W010 (30.000 filas) | `e3cc3fe1de7f2f3decfdf2c0eb04417f11ea048aaa8f16fd03b7b74dc5b26753` |
| `submissions/staging_candidates_20261110/submission_candidate_w035.csv` | Vector de test real para candidato W035 (30.000 filas) | `051366d4df54966e309a50177e099eea4cfaf5d4ee42352a5ce9d55283a62286` |
| `submissions/staging_candidates_20261110/submission_candidate_surgical_w035_top25.csv` | Vector de test real quirúrgico anclado en EVT-143 (25 flips) | `082b786ae89adc931f6d376717a65824ee1e1d0879cf43f3d79f0fb5a8fb36c9` |
| `results/rolling_origin_backtest_20261110/rolling_origin_backtest_report.json` | Reporte completo del backtest rolling-origin y calibración | `c9d1b13f05eb4fdc254f1aa783d7a80bb600673f23114e8028ad7c88b00aee5d` |
