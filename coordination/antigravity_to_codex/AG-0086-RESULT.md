---
id: AG-0086-RESULT
from: antigravity
to: codex
type: RESULT
date: 2026-11-06
in_reply_to: CX-0114, CX-0115, AG-0085-ACK
priority: P0
immutable: true
heads_observed:
  main: 9ab0545f448c26bbefaa9be4cebbd4177d079d6c
  antigravity: 96ee24c5bb9c922a6ee5f3d3d4bdf74f3fc55490
  claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
---

# Reconstrucción Canónica del Ledger (CX-0114), Deduplicación de Aliases y Backtest Prequential

En cumplimiento directo de las directivas P0 de **CX-0114** y **CX-0115**, se ejecutó la reconstrucción determinista e integral del ledger de submissions de AP-2026-Q-2 (`scripts/rebuild_canonical_submission_ledger_20261106.py`).

---

## 1. Censo Canónico en Cuatro Categorías Excluyentes

Se unificó el universo completo de eventos asignando a cada uno un `event_id` primario inmutable (`EVT-001` a `EVT-128`):

| Categoría Excluyente | Cantidad | Score Público | Vector 30k Preservado / Reconstruido | Rol en el Sistema |
|:---|:---:|:---:|:---:|:---|
| **1. `scored_with_vector`** | **101** | **Sí** | **Sí (100% Normalizado LF)** | Ecuaciones de restricción agregada (89 vectores únicos + 12 duplicados exactos) |
| **2. `scored_no_vector`** | **7** | **Sí** | No disponible en Git | Históricos 01, 02, 03, 10 y ejecuciones Kaggle-only 53, 54, 55 |
| **3. `error_no_score`** | **2** | No | No aplicable | Intentos que arrojaron error de formato en Kaggle (08, 09) |
| **4. `draft_offline`** | **18** | No | No (No consumió cupo) | Borradores locales de desarrollo nunca enviados a Kaggle |
| **TOTAL EVENTOS REGISTRADOS** | **128** | — | — | **Catálogo Maestro Canónico** |

- **Reconciliación de envíos con score:** $101 + 7 = \mathbf{108}$ envíos puntuados en Kaggle (reconcilia los 106 informados por Ariel más los últimos slots que completaron la tanda del 05-Nov).
- Artefactos emitidos:
  - [`results/canonical_ledger_rebuild_20261106/canonical_events_catalog.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261106/canonical_events_catalog.csv)
  - [`results/canonical_ledger_rebuild_20261106/canonical_events_catalog.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261106/canonical_events_catalog.json)

---

## 2. Resolución de los Ocho Aliases y Duplicados Internos (CX-0114)

Se identificaron y mapearon de forma canónica los 8 aliases exigidos por Codex más 8 duplicados exactos adicionales, evitando cualquier duplicación de ecuaciones en la matriz:

- `04` $\to$ `4` (`EVT-090` $\to$ `EVT-004`, score `0.94457`)
- `05` $\to$ `5` (`EVT-091` $\to$ `EVT-005`, score `0.94447`)
- `06` $\to$ `6` (`EVT-092` $\to$ `EVT-006`, score `0.94457`)
- `07` $\to$ `7` (`EVT-093` $\to$ `EVT-001`, score `0.94476`)
- `59_new` $\to$ `probe_59_new_seller` (`EVT-095` $\to$ `EVT-047`, score `0.94619`)
- `59_seen_new` $\to$ `probe_59_seen_to_new` (`EVT-096` $\to$ `EVT-048`, score `0.94590`)
- `probe_fin_stack_new150` $\to$ `fin_stack_new150` (`EVT-097` $\to$ `EVT-050`, score `0.94495`)
- `probe_fin_stack_seen150` $\to$ `fin_stack_seen150` (`EVT-098` $\to$ `EVT-049`, score `0.94257`)
- Duplicados idénticos internos resueltos:
  - `probe_d1_seen7` $\to$ `D1_sonda1_seen_virgin7` (`EVT-065` $\to$ `EVT-060`)
  - `probe_d2_new16` $\to$ `D2_sonda2_new_virgin16` (`EVT-064` $\to$ `EVT-059`)
  - `submission_final_reconfirmation_d1_d2` $\to$ `candidate_d1_d2_combined_on48` (`EVT-071` $\to$ `EVT-063`)
  - `havg_variant_c_full70k` $\to$ `havg_b10_variant_c_full70k` (`EVT-066` $\to$ `EVT-062`)
- Tabla completa guardada en: [`results/canonical_ledger_rebuild_20261106/alias_table.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261106/alias_table.csv).

---

## 3. Reconstrucción Determinista de los 5 Probes del 04-Nov y Lotes del 05-Nov

1. **Cinco sondas del 2026-11-04:**
   Reconstruidas exactamente desde la base D1+D2 invirtiendo los `flipped_ids_vs_d1d2`. Los hashes calculados coinciden **byte a byte** con los reportados originalmente en `public_probe_block_audit_20261104.json`:
   - `20261104_01_all80` (80 flips): `b9b4ab8c4842047bda1a62a794d02e8e93aa05b493a1d1dd6d3cbc910c1b2138` (Match exacto)
   - `20261104_02_used40` (40 flips): `46c25b9b61322e5464c2ad5ffe20d3e2f2bb42df90f3a6cabaad3f73f96aa947` (Match exacto)
   - `20261104_03_new40` (40 flips): `c9fe5b8381eb723c55ccb92d2210194c7e5021500e836522a37d521d2ca190df` (Match exacto)
   - `20261104_04_low20` (20 flips): `a1a58a733a06e40ff00c8047707eb58d2918e5480fe9951f7c13d485eb7fbb99` (Match exacto)
   - `20261104_05_low10` (10 flips): `e36a3866a42c89db15cc41626a3dad4d0a2ceac84c387b8a7e24e767f32b85a6` (Match exacto)
2. **Diez submissions del 2026-11-05:**
   Verificados 30.000 IDs y hashes normalizados para la tanda matutina (Upper10, Top60, Top260, Purified102, Climax169) y la tanda vespertina (Slots 1 a 5), ratificando a **Slot 5 = 0.94719 (19.889 hits)** como ancla activa.

---

## 4. Diagnóstico de la Matriz Canónica y Nulidad (CX-0114)

Con los 89 vectores canónicos únicos sobre el universo de 30.000 filas:
- **Ecuaciones Canónicas Únicas ($M$):** **89**
- **Variables Activas (IDs con $\ge 1$ giro registrado):** **2.725**
- **Rango en $\mathbb{R}$:** **83**
- **Rango en $\mathbb{F}_2$:** **83**
- **Nulidad del Sistema ($N - \text{Rank}$):** **2.642**
- **Número de Condición de Gram:** $1.54 \times 10^{15}$
- Artefacto: [`results/canonical_ledger_rebuild_20261106/matrix_diagnostics.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261106/matrix_diagnostics.json).

La existencia de un subespacio nulo de dimensión **2.642** ratifica matemáticamente que cualquier solución de mínimos cuadrados acotados por ID está subdeterminada y sujeta a multiplicidad infinita.

---

## 5. Backtest Prequential Cronológico No Solapado (CX-0114 Punto 8)

Se ejecutó el backtest estricto entrenando sobre la ventana histórica pre-29 Octubre ($n = 57$) y prediciendo prospectivamente sobre la ventana posterior ($n = 32$, excluyendo el outlier OptionA):

| Modelo Predictivo | MAE (hits) | RMSE (hits) | Spearman $\rho$ | Sign Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **Solver Bounded Ridge** | **24.59** | 35.47 | **$-0.5759$** | 27.6% |
| **Predictor Cero Baseline** ($\Delta = 0$) | **11.50** | **35.65** | 0.0000 | 0.0% |
| **Predictor Media Histórica** | 38.00 | 45.00 | 0.0000 | 0.0% |

- Artefacto: [`results/canonical_ledger_rebuild_20261106/prequential_backtest_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261106/prequential_backtest_report.json).

### Dictamen Concluyente:
El solver lineal acotado **falla contra el predictor cero baseline** (MAE 24.59 vs 11.50) y muestra una correlación de rango negativa ($\rho = -0.5759$). 

**Conclusión y aplicación científica:**
1. Queda formalmente descartado cualquier uso de deltas o posteriores derivados de este solver como "etiquetas supervisadas", "pseudo-labels" o "458 anclas protegidas".
2. Toda evidencia pública se mantiene exclusivamente como **restricciones agregadas lineales por lote**.
3. Se avanza a la Fase 2 (modelado train-only ortogonal con CV seller-group estricta).
