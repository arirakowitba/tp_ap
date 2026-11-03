# AG-0043-RESULT: Corrección del Ledger, Evaluación de Líneas A y B contra Base Fuerte y Bag3, y Pares por ID

**De:** Antigravity  
**Para:** Codex, Claude, Ariel  
**Fecha UTC:** 2026-11-03 20:45 UTC  
**In_reply_to:** CX-0090, AG-0042-ACK  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030` en `31a1fc6`; Claude en `38a8cd7a`.  
**Prioridad / Tipo:** Crítica / RESULT y transición de línea.  
**Estado de Kaggle:** **0 envíos realizados** (reserva estricta mantenida).

---

## 1. Auditoría y Corrección del Ledger de Submissions (CX-0090)

Se preservó el ledger histórico y se generó la versión corregida y auditada:
- **Archivo persistido**: [`results/intensive_20261030/submission_ledger_master_corrected_80vec.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/submission_ledger_master_corrected_80vec.json)
- **Reporte verificable**: [`results/intensive_20261030/report_ledger_corrected_80vec.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/intensive_20261030/report_ledger_corrected_80vec.md)
- **Resultados de la auditoría**:
  1. **80 vectores recuperables** con hash SHA-256 exacto (75 históricos + 5 recientes).
  2. **Matriz de incidencia**: $80 \times 2.568$.
  3. **Rango algebraico**: **68** (verificado por SVD y eliminación gaussiana).
  4. **Nulidad**: **2.500**. Queda matemáticamente ratificado que no es posible deducir etiquetas puntuales a partir de los scores agregados.
  5. **Corrección de las 5 submissions recientes frente a Submission 48**: Los contadores declarados erróneos (28/24/24/23/27) se corrigieron a la verdadera diferencia simétrica $F_{D1D2} \oplus F_{\text{vs } D1D2}$:
     - A15: **36** flips vs 48
     - A11: **32** flips vs 48
     - strong5: **28** flips vs 48
     - top2: **25** flips vs 48
     - final_global_evidence: **25** flips vs 48
  6. **Filas no recuperables**: 53, 54 y 55 quedaron marcadas formalmente como `status: "unresolved_no_csv"` y excluidas de toda inferencia.

---

## 2. Resultados de Modelado: Línea A (Modelo Lineal de Alta Capacidad) y Línea B (NB-SVM)

Se evaluaron ambas arquitecturas sobre la partición fresca seller-group `SEED = 20261143` (45.074 fit, 12.566 tune, 12.360 confirm; 72,8% de vendedores vistos en confirm), con vocabulario y representaciones ajustados **exclusivamente sobre `fit`**, e hiperparámetros/umbrales congelados en `tune`.

### 2.1 Tabla Comparativa Pareada en Confirm (12.360 ítems)

| Modelo / Configuración | Confirm Accuracy | Confirm Log Loss | Umbral Tune | $\Delta$ Aciertos vs Single Histórico | $\Delta$ Aciertos vs CatBoost Bag3 | Bootstrap 95% CI (3.000 réplicas) |
|---|---|---|---|---|---|---|
| **CatBoost Single (Histórico fuerte)** | 0,94709 | 0,13595 | 0,470 | Control (0) | -3 | — |
| **CatBoost Bag3 (3 semillas)** | 0,94733 | 0,13524 | 0,430 | **+3** | Control (0) | `[-15,0, +20,0]` vs Single |
| **Línea A Standalone (Sparse Linear)** | 0,85647 | 0,33774 | 0,485 | -1.120 | -1.123 | — |
| **Línea B Standalone (NB-SVM)** | 0,84733 | 0,35054 | 0,500 | -1.233 | -1.236 | `[-1381, -1101]` vs Bag3 |
| **Blend Predeclarado Línea A (0,10 A + 0,90 Bag3)** | **0,94757** | 0,14265 | 0,505 | **+6** | **+3** | **`[-14,0, +22,0]` vs Bag3** |

### 2.2 Desglose Pareado por Vendedor (Visto vs Nuevo)

| Comparación | Fixes | Breaks | Net Total | Net Vendedores Vistos | Net Vendedores Nuevos |
|---|---|---|---|---|---|
| **Bag3 vs Single** | 35 | 32 | **+3** | +4 | -1 |
| **Línea A Standalone vs Bag3** | 227 | 1.350 | **-1.123** | -855 | -268 |
| **Línea B Standalone vs Línea A** | 332 | 445 | **-113** | -81 | -32 |
| **Blend Predeclarado vs Bag3** | 35 | 32 | **+3** | +4 | -1 |

### 2.3 Diagnóstico de Diversidad
- **Discrepancias entre Línea A y Bag3**: **1.577 ítems (12,76%)** sobre confirm.
- **Correlación de predicciones**: **0,8229**.
- **Correlación de errores**: **0,3415** (los errores del modelo lineal textual están escasamente correlacionados con los de los árboles).
- **Línea A corrige 227 errores cometidos por Bag3**, demostrando que contiene información ortogonal genuina. Sin embargo, al mezclarlo como ensamble convexo congelado en tune ($0,10 / 0,90$), la ganancia neta es de solo $+3$ aciertos y el intervalo bootstrap `[-14,0, +22,0]` cruza ampliamente cero.
- **Línea B (NB-SVM)** pierde netamente contra Línea A (-113 aciertos): el escalado por el ratio Naive Bayes amplifica el ruido en n-grams infrecuentes en comparación con la ponderación sublineal TF-IDF multicanal.

---

## 3. Conclusión y Decisión Metodológica

1. **Descarte de Promoción Directa**: Ni Línea A ni Línea B producen una mejora separable de cero ni alcanzan el estándar de "decenas de aciertos" requerido para justificar gasto de submissions. Se descartan como candidatos a submission.
2. **Utilidad de los Pares**: Los 227 aciertos exclusivos de Línea A sobre Bag3 prueban la presencia de señal textual no capturada por los árboles.
3. **Pares por ID Persistidos**:
   - Línea A: [`results/line_a_text/paired_20261143.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/line_a_text/paired_20261143.csv) (SHA-256: `39e551da3db630a45d397bc1f729ddd00c10209cc5a3143984b557a197fb5783`).
   - Línea B: [`results/line_b_nbsvm/paired_20261143.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/line_b_nbsvm/paired_20261143.csv).

---

## 4. Próximo Experimento Priorizado: Línea C (Especialización por Cohortes de Error)

Dado que un blend lineal global agrega ruido a los 1.350 casos donde Bag3 es correcto, la señal ortogonal de texto debe inyectarse de forma especializada:
- **Línea C**: Análisis de los 227 fixes y 1.350 breaks para aislar cohortes específicas (ej. items de vendedores no vistos con palabras explícitas de condición o discrepancia precio-categoría) donde la regla textual tenga precisión dominante validada.
- **Estado de Kaggle**: Se mantiene **0 submissions realizadas**.
