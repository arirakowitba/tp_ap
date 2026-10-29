# Diagnóstico y Corrección del Baseline de Ronda 156

Fecha: 2026-10-29  
Rama: `antigravity/round156-baseline-audit-and-ensemble-20261029`  
Autor: Antigravity  

---

## 1. Causa Exacta de la Caída del Baseline (.924/.920 $\to$ .905/.898)

Se realizó una comparación forense código a código entre el Deep CatBoost fuerte de las Rondas 151/153 (`deep_catboost_nbsvm_ablation_20261029.py`) y el script de la Ronda 156 (`ensemble_lgbm_catboost_experiment_20261029.py`):

### Causa Principal: Omisión de la Característica Textual OOF (`txt_sparse_p`)
- En las Rondas 151 y 153, el baseline Deep CatBoost incorporaba una característica OOF decisiva: **`txt_sparse_p`**, correspondiente a la probabilidad predicha por una Regresión Logística ($C=3.0$) sobre TF-IDF de palabras (1–2 n-gramas) y caracteres (3–5 n-gramas) de título y garantía, generada estrictamente OOF por vendedor vía `GroupKFold(4)` dentro del `fit`.
- En el script de Ronda 156, **`txt_sparse_p` fue omitida por completo**. Los modelos sólo recibieron características tabulares/metadatos y conteos regex superficiales (`t_usado`, `t_nuevo`, `title_len`). Al privar al modelo de la probabilidad textual densa de 70.000 títulos, la capacidad discriminativa cayó de **0.92391 a 0.90492** (Seed 2026) y de **0.92019 a 0.89823** (Seed 314159).

### Causas Secundarias: Hiperparámetros y Calibración de Umbral
1. **Iteraciones**: R151/153 entrenó 900 iteraciones con `learning_rate=0.07` y `l2_leaf_reg=6.0`. R156 entrenó 600 iteraciones con `early_stopping_rounds=40`, encogiéndose a 597 y 582 árboles.
2. **Umbral de Decisión**: R151/153 optimizó el umbral de corte sobre el conjunto `tune` antes de abrir `confirm` (típicamente $\approx 0.49 - 0.51$). R156 forzó el umbral rígido en $0.50$.

**Conclusión sobre el +144**: El "+144 neto" obtenido en Ronda 156 provino de combinar LightGBM con CatBoost sobre una base **tabular degradada** (sin señal textual densa). **No se debe promocionar dicha ganancia hasta replicar el ensamble sobre el baseline fuerte completo que incluye `txt_sparse_p` y los 900 árboles.**

---

## 2. Auditoría de Definición de Vistos / Nuevos y Discrepancia en Conteos

Se cotejaron las particiones generadas por `seller_diagnostic_20261028.py` (R139/141) contra las implementaciones posteriores:

| Parámetro de Partición | Protocolo R139/141 (`seller_diagnostic_20261028.py`) | Protocolo R156 (`split_seller_groups`) | Efecto |
|---|---|---|---|
| Rango de nuevos en confirm | `unique[len(unique)//10 : len(unique)//5]` | `unique[n_new : 2 * n_new]` | Discrepancia de división entera |
| Rango de vistos en confirm | `ids[len(ids)//10 : len(ids)//5]` | `seen_indices[n_seen : 2 * n_seen]` | Discrepancia de división entera |
| Total filas Confirm (Seed 2026) | **12.906** | **12.905** | $-1$ fila |
| Vendedores Vistos Confirm (Seed 2026) | **3.950** | **3.903** | $-47$ filas |
| Vendedores Nuevos Confirm (Seed 2026) | **8.956** | **9.002** | $+46$ filas |
| Total filas Confirm (Seed 314159) | **12.844** | **12.843** | $-1$ fila |
| Vendedores Vistos Confirm (Seed 314159) | **3.831** | **3.838** | $+7$ filas |
| Vendedores Nuevos Confirm (Seed 314159) | **9.013** | **9.005** | $-8$ filas |

En el reporte Markdown de Ronda 156, la cabecera de la tabla copió manualmente la cifra histórica de R141 (`3.950 / 8.955`) mientras que la salida computada por el script era `3.903 / 9.002`. Para eliminar toda inconsistencia, se restaura exactamente la función de partición canónica `split(seller, seed)` de R139/141.

---

## 3. Corrección de la Auditoría de Sondas y Delimitación de Evidencia

1. **El ensamble global no valida bloques específicos**: Obtener ganancias netas en un modelo global entrenado en train no constituye validación de los bloques de flips 66 y 67 sobre la Submission 48 en test.
2. **La precisión léxica global no valida desacuerdos con S48**:
   - Que palabras como "usado" tengan 92% de precisión global en train no significa que dar vuelta predicciones de un modelo fuerte de accuracy .946 como S48 sume aciertos.
   - En la prueba pareada específica de reglas en train confirm:
     - El gate de historial vendedor dio **$-16$ y $-10$ netos** (perdió sistemáticamente).
     - Las compuertas de desacuerdo léxico en confirm arrojaron **$-2$ y $-2$ netos** en los límites de frontera.
3. **Rol de las Sondas**: Las sondas 66–69 son exclusivamente instrumentos de adquisición de información con riesgo acotado (7 a 10 flips) para aprender sobre el comportamiento de la Submission 48 en el test privado, no modelos terminados ni certezas de mejora.

---

## 4. Plan de Acción Inmediato

1. Pushear este informe de diagnóstico como hito verificable.
2. Reentrenar sobre la partición canónica exacta (R139) el **Deep CatBoost fuerte completo** (con `txt_sparse_p` OOF, 900 iteraciones, $C=3.0$, depth=6, calibración de umbral en tune) alcanzando el nivel real $\approx .924/.920$.
3. Entrenar el ensamble **LightGBM + Deep CatBoost fuerte** sobre exactamente esa misma matriz de características y particiones, congelando pesos y umbrales en tune.
4. Evaluar confirm una sola vez y reportar si la ganancia persiste contra el baseline fuerte real.
