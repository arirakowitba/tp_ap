# AG-0045-RESULT: Auditoría de Cohortes, Evaluación de Mezcla Predeclarada Bag3 + NB-SVM en Corte Fresco y Modelo de Campos Prefijados

**ID:** AG-0045-RESULT  
**Fecha UTC:** 2026-11-03 21:30 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** CX-0091, AG-0044-ACK  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030` (`42166f0` integrando `origin/main` en `647dfc89`); Claude en `38a8cd7a`.  
**Prioridad / Tipo:** Alta / RESULT experimental y cierre de líneas de ensamble.  
**Estado de Kaggle:** **0 envíos realizados** (50/50 cuotas preservadas al 100%).

---

## 1. Auditoría Completa de la Discrepancia de Cohortes (227 Fixes / 1.350 Breaks)

En respuesta a la auditoría requerida sobre el reporte de cohortes (`report_line_c_cohort_analysis_20261143.json`), se implementó y ejecutó [`scripts/audit_line_c_cohort_discrepancy.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/audit_line_c_cohort_discrepancy.py) con trazabilidad completa sobre los 12.360 registros de `confirm` del split `20261143` ([`results/line_a_text/paired_20261143.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/line_a_text/paired_20261143.csv), SHA-256: `39e551da3db630a45d397bc1f729ddd00c10209cc5a3143984b557a197fb5783`).

### 1.1 Identificación Exacta de los Modelos
- **Modelo A (Evaluado)**: `Línea A Standalone` (regresión logística regularizada sobre TF-IDF multicanal sparse de 171.459 variables: título n-gramas 1-3 palabra y 3-6 char, garantía y atributos). **No contiene `seller_id`, precio, categoría, ciudad ni metadata tabular**.
  - Accuracy en confirm: **0,85647** (10.586 aciertos / 12.360).
- **Modelo B (Referencia)**: `CatBoost Bag3` (promedio de 3 semillas con 101 features tabulares profundas + `seller_id` como categoría nativa + target encodings suavizados + texto OOF supervisado).
  - Accuracy en confirm: **0,94733** (11.709 aciertos / 12.360).

### 1.2 Matriz de Contingencia 2x2 en Confirm
| | CatBoost Bag3 Correcto (11.709) | CatBoost Bag3 Incorrecto (651) | Total Línea A |
|---|---|---|---|
| **Línea A Correcto** | 10.359 (Ambos aciertan) | **227 (Fixes de Línea A)** | 10.586 |
| **Línea A Incorrecto** | **1.350 (Breaks de Línea A)** | 424 (Ambos fallan) | 1.774 |
| **Total Bag3** | 11.709 | 651 | 12.360 |

### 1.3 Conclusiones de la Auditoría
1. **Ausencia de bugs de inversión o polaridad**: El saldo de $-1.123$ ($227 - 1.350$) no es un error de código ni polaridad invertida; es la consecuencia matemática esperable de comparar un modelo de texto puro con $85,6\%$ de accuracy contra un ensamble multimodal profundo de árboles con $94,7\%$ de accuracy.
2. **Ortogonalidad genuina**: Los **227 fixes** demuestran que el texto contiene señal discriminativa en casos donde los árboles se confunden por el contexto de vendedor o atributos tabulares.
3. **Falla del Confidence Router**: Un filtro ingenuo por margen ($|p_{\text{text}} - 0,5| > 0,20$ en la zona de incertidumbre del árbol $|p_{\text{bag3}} - 0,5| < 0,10$) arrojó **28 fixes vs 29 breaks (saldo neto -1)**. El margen lineal sin contexto no aísla los fixes de los breaks. Reporte completo archivado en [`results/line_c_cohorts/audit_cohort_discrepancy_verified.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/line_c_cohorts/audit_cohort_discrepancy_verified.md).

---

## 2. Experimento Prioritario: CatBoost Bag3 + NB-SVM Predeclarado en Split Fresco (20261144)

Siguiendo la directiva de CX-0091, se evaluó una mezcla de `CatBoost Bag3` con un `NB-SVM` independiente entrenado en un corte seller-group completamente fresco (`SEED = 20261144`, disjunto de 2026, 314159 y 20261143).
- **Tamaño confirm**: 12.264 filas (73,4% vendedores vistos, 26,6% nuevos).
- **Protocolo estricto**: Ponderación $\alpha$ y umbral de decisión seleccionados exclusivamente en `tune` y congelados antes de evaluar en `confirm`.
- **Script**: [`scripts/train_nbsvm_bag3_predeclared_blend.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_nbsvm_bag3_predeclared_blend.py).
- **Archivo de pares**: [`results/nbsvm_bag3_blend/paired_20261144.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/nbsvm_bag3_blend/paired_20261144.csv) (SHA-256: `10cd856ace7c5ab7b0f00dba37d0dfc1cdc2d0f74833ff1fe870aa7de9b77480`).

### 2.1 Resultados Pareados en Confirm (20261144)

| Modelo / Configuración | Confirm Acc | Confirm LogLoss | Umbral (tune) | Delta vs Single Histórico | Delta vs Bag3 | Bootstrap 95% CI (3.000 réplicas) |
|---|---|---|---|---|---|---|
| **CatBoost Single Histórico** | 0,95360 | 0,12290 | 0,485 | Control (0) | -21 | — |
| **CatBoost Bag3** | 0,95532 | 0,12035 | 0,515 | **+21** | Control (0) | `[+5.0, +38.0]` vs Single |
| **NB-SVM Standalone** | 0,84915 | 0,34395 | 0,410 | -1.281 | -1.302 | — |
| **Blend Predeclarado ($\alpha=0.05$)** | **0,95581** | **0,12419** | 0,530 | **+27** | **+6** | **`[-3.0, +15.0]` vs Bag3** |

### 2.2 Desglose Pareado por Tipo de Vendedor
| Comparación | Fixes | Breaks | Net Total | Net Vistos (73,4%) | Net Nuevos (26,6%) |
|---|---|---|---|---|---|
| **Bag3 vs Single** | 45 | 24 | **+21** | +5 | +16 |
| **NB-SVM Standalone vs Bag3** | 212 | 1.514 | **-1.302** | -966 | -336 |
| **Blend Predeclarado vs Bag3** | 13 | 7 | **+6** | +3 | +3 |

### 2.3 Criterio de Decisión (CX-0091)
- La mezcla predeclarada produce apenas 20 discrepancias con Bag3 (0,16%).
- Aunque el saldo puntual es de $+6$ aciertos netos, el intervalo bootstrap del 95% **cruza claramente el cero** (`[-3.0, +15.0]`) y el log loss **empeora** de 0,12035 a 0,12419 debido a la mala calibración de la probabilidad lineal extrema.
- **Decisión**: Conforme al mandato expreso de CX-0091 («*Si la mezcla no logra mejora pareada estable de decenas o IC claramente positivo, cerrar línea y pasar a modelo lineal sparse de campos prefijados/atributos clave-valor*»), **la línea de mezclas post-hoc Bag3 + NB-SVM queda formally CERRADA**.

---

## 3. Prioridad 2: Modelo Lineal Sparse de Campos Prefijados (Línea A2)

Se implementó y evaluó [`scripts/train_prefixed_fields_sparse_model.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_prefixed_fields_sparse_model.py) sobre el mismo corte fresco `20261144`:
- **Representación**:
  - Canal título: n-gramas de palabras 1-3 y caracteres 3-6 con normalización de unidades (`128_gb`, `0km` -> `cerokm`).
  - Canal garantía: n-gramas 1-2.
  - Canal estructurado prefijado: pares `aid_<id>=<val>`, `name_<name>=<val>`, y tokens de valor.
  - Dimensión total del vocabulario en fit: **161.287 variables**.
- **Resultados Standalone en Confirm 20261144**:
  - **Confirm Accuracy**: **0,85731** (10.514 correctos / 12.264).
  - **Confirm Log Loss**: **0,33580**.
  - **Comparación con NB-SVM Standalone**: Mejora de **+100 aciertos netos** en confirm (NB-SVM obtuvo 0,84915 / LL 0,34395).
  - Artefactos guardados en [`results/prefixed_sparse_model/report_prefixed_sparse_20261144.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/prefixed_sparse_model/report_prefixed_sparse_20261144.json) y array de probabilidades `confirm_probs_20261144.npz`.

---

## 4. Estado de Conciliación del Ledger y Cuota Kaggle

1. **Ledger Maestro Corregido**:
   - Conforme a CX-0090 y `scripts/audit_submission_ledger_rank_20261103.py`, se completó [`results/intensive_20261030/submission_ledger_master_corrected_80vec.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/submission_ledger_master_corrected_80vec.json):
     - 80 vectores exactos con CSV y hash SHA-256.
     - Filas 53, 54 y 55 marcadas `unresolved_no_csv`.
     - Flips de las 5 submissions recientes verificados por diferencia simétrica vs Sub 48: A15 (36), A11 (32), strong5 (28), top2 (25), final_global_evidence (25).
     - Rango algebraico: **68**. Nulidad: **2.500** sobre los 2.568 IDs intervenidos.
     - Se respeta estrictamente el principio de no derivar labels individuales ni inventar imputaciones.
2. **Cuota Kaggle**:
   - **0 submissions consumidas**. Se mantienen las 50 submissions disponibles para la fase final que autorice Ariel.

---

## 5. Próximo Paso Técnico

Habiendo cerrado la mezcla lineal post-hoc por falta de significancia estadística, y habiendo verificado que los campos prefijados superan al texto simple por +100 aciertos standalone:
- **Hipótesis siguiente**: En lugar de promediar linealmente probabilidades mal calibradas, generar la probabilidad OOF de los campos prefijados (`oof_p_prefixed`) vía GroupKFold por vendedor sobre `fit` e inyectarla directamente como una feature continua en el pool tabular de `CatBoost`. Esto permite que los árboles utilicen la señal textual condicionada por vendedor y metadatos, evitando los falsos positivos lineales.
