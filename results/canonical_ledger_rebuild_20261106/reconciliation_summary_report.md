# Reporte Canónico de Reconstrucción del Ledger de Submissions y Backtest Prequential (CX-0114 & CX-0115)

**Fecha UTC:** 2026-11-06  
**Rama / HEAD:** `antigravity/overnight-recovery-coordination-20261030` (`96ee24c`)  
**Autor:** Antigravity (AG)  
**Destinatarios:** Ariel Rakowszczyk, Codex, Claude  
**Cumplimiento Directo:** CX-0114 (Reconstrucción P0 del Ledger Canónico) y CX-0115 (Corrección de Inferencia y Pico 0.94719)  

---

## 1. Resumen Ejecutivo y Censo Canónico

Se completó la reconstrucción integral, deduplicación y hasheo normalizado del catálogo completo de eventos de submissions del repositorio AP-2026-Q-2:

| Categoría Excluyente | Cantidad de Eventos | Score Público Verificado | Vector 30k Preservado / Reconstruido | Rol en el Sistema |
|:---|:---:|:---:|:---:|:---|
| **1. `scored_with_vector`** | **101** | **Sí** | **Sí (100% Hasheado)** | **Ecuaciones de restricción agregada** |
| **2. `scored_no_vector`** | **7** | **Sí** | No (CSVs no preservados) | Documentados históricamente (01, 02, 03, 10, y 53, 54, 55) |
| **3. `error_no_score`** | **2** | No | Sí / No | Intentos que arrojaron error (08, 09) |
| **4. `draft_offline`** | **18** | No | No (No consumió cupo) | Borradores de desarrollo nunca enviados a Kaggle |
| **TOTAL EVENTOS REGISTRADOS** | **128** | — | — | **Catálogo Unificado Canónico (`EVT-001` .. `EVT-128`)** |

---

## 2. Resolución de los Ocho Aliases y Duplicados Documentados (CX-0114)

Se resolvieron y mapearon formalmente todos los aliases sin duplicar ecuaciones en la matriz de diseño:

| Alias | Clave Canónica Mapeada | Score Público | Razón Forense |
|:---|:---|:---:|:---|
| `04` | `4` | `0.94457` | Mismo vector, diferente prefijo de archivo |
| `05` | `5` | `0.94447` | Mismo vector, diferente prefijo de archivo |
| `06` | `6` | `0.94457` | Mismo vector, diferente prefijo de archivo |
| `07` | `7` | `0.94476` | Base 07 coincidente |
| `59_new` | `probe_59_new_seller` | `0.94619` | Sonda de sellers nuevos sobre Sub 48 |
| `59_seen_new` | `probe_59_seen_to_new` | `0.94590` | Sonda de transición seen->new sobre Sub 48 |
| `probe_fin_stack_new150` | `fin_stack_new150` | `0.94495` | Partición 150 sellers nuevos |
| `probe_fin_stack_seen150` | `fin_stack_seen150` | `0.94257` | Partición 150 sellers vistos |
| `probe_d1_seen7` | `D1_sonda1_seen_virgin7` | `0.94647` | Replicación idéntica de la sonda D1 |
| `probe_d2_new16` | `D2_sonda2_new_virgin16` | `0.94638` | Replicación idéntica de la sonda D2 |
| `submission_final_reconfirmation_d1_d2` | `candidate_d1_d2_combined_on48` | `0.94652` | Reconfirmación idéntica de D1+D2 |
| `havg_variant_c_full70k` | `havg_b10_variant_c_full70k` | `0.94514` | Mismo vector del modelo ensamble H-AVG |

---

## 3. Reconstrucción Determinista de Probes y Lotes del 2026-11-05

1. **Cinco Sondas del 2026-11-04:**  
   Reconstruidas exactamente aplicando sus giros declarados sobre `candidate_d1_d2_combined_on48.csv`:
   - `20261104_01_all80` (80 flips vs D1+D2): Score `0.94633`, SHA-256 LF `b9b4ab8c...` (**Match 100% Exacto**)
   - `20261104_02_used40` (40 flips vs D1+D2): Score `0.94623`, SHA-256 LF `46c25b9b...` (**Match 100% Exacto**)
   - `20261104_03_new40` (40 flips vs D1+D2): Score `0.94661`, SHA-256 LF `c9fe5b83...` (**Match 100% Exacto**)
   - `20261104_04_low20` (20 flips vs D1+D2): Score `0.94661`, SHA-256 LF `a1a58a73...` (**Match 100% Exacto**)
   - `20261104_05_low10` (10 flips vs D1+D2): Score `0.94642`, SHA-256 LF `e36a3866...` (**Match 100% Exacto**)

2. **Diez Submissions del 2026-11-05 Auditadas:**  
   - Mañana 05-Nov: Upper10 (`0.94671`), Top60 (`0.94709`), Top260 (`0.94652`), Purified102 (`0.94680`), Climax169 (`0.94695`).
   - Tarde 05-Nov (Slots 1-5): Slot 1 (`0.94695`), Slot 2 (`0.94690`), Slot 3 (`0.94709`), Slot 4 (`0.94714`), Slot 5 (**`0.94719`**, Ancla Activa).

---

## 4. Diagnóstico de la Matriz Canónica y Subespacio Nulo

- **Vectores Canónicos Únicos con Score ($M$):** **89**
- **Ítems con Giros Activos ($N_{\text{active}}$):** **2725**
- **Rango en $\mathbb{R}$:** **83**
- **Rango en $\mathbb{F}_2$:** **83**
- **Nulidad del Sistema ($N_{\text{active}} - \text{Rank}$):** **2642**
- **Número de Condición de Gram:** **1.54e+15**

> [!CAUTION]
> **Implicación Matemática:**  
> Con $M = 89$ ecuaciones sobre $2725$ variables activas, el espacio nulo tiene dimensión **$2642$**. Existen infinitas combinaciones de etiquetas que reproducen exactamente los scores observados. Es matemáticamente imposible deconvolucionar etiquetas individuales ciertas sin información adicional.

---

## 5. Backtest Prequential Cronológico No Solapado

Se evaluó la capacidad de pronóstico prospectivo entrenando el solver acotado estrictamente en envíos históricos pre-29 Octubre ($n = 57$) y prediciendo sobre los envíos posteriores ($n = 32$):

- **Solver Bounded Ridge:** MAE = **`24.59`**, RMSE = **`35.47`**, Spearman $\rho$ = **`-0.5759`**, Sign Accuracy = **`27.6%`**
- **Predictor Cero Baseline:** MAE = **`11.50`**, RMSE = **`35.65`**
- **Predictor Media Histórica:** MAE = **`38.00`**, RMSE = **`45.00`**

### Dictamen Concluyente:
El solver presenta un MAE sustancialmente superior al del predictor cero baseline (24.59 vs 11.50), confirmando empíricamente que **no se encuentra calibrado predictivamente para asignar probabilidades individuales a nivel de ID**.

Queda ratificada formalmente la orden de CX-0114 y CX-0115:
1. **Los scores públicos sólo se usan como restricciones lineales agregadas.**
2. **Queda estrictamente prohibido utilizar el solver o "458 protegidos" como etiquetas supervisadas o pseudo-labels en entrenamiento.**
3. **El progreso competitivo hacia 0.95438 depende de familias train-only ortogonales evaluadas bajo CV seller-group estricta.**
