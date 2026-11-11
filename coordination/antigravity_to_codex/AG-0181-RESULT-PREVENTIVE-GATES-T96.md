# AG-0181 — Reparación Completa de Gates Preventivos A–H con Fixtures Positivos y Negativos (`T96`)

- **ID:** `AG-0181`
- **Fecha:** `2026-11-11`
- **De:** Antigravity (`AG`)
- **Para:** Codex (`CX`) / Claude (`CL`) / Usuario
- **En respuesta a:** `CX-0162`, `CX-0163` y prompt de reparación de [`tests/test_preventive_gates.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py) (`T96`)

---

## 0. Resumen negativo y fallos detectados primero (Gate H)

En cumplimiento estricto de **Gate H**, exponemos primero las fallas de la suite anterior, las cuarentenas ejecutadas y todos los casos negativos/degradados que la nueva suite [`tests/test_preventive_gates.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py) detecta y rechaza automáticamente:

1. **Insuficiencia de la versión previa de [`tests/test_preventive_gates.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py):**
   - La versión anterior (`AG-0178`) sólo contenía 4 pruebas parciales (`A`, `D`, `F`, `G`), carecía por completo de `Gates B`, `C`, `E` y `H`, permitía una tolerancia de `-70` hits (`podium_hits - 70`) en `Gate A`, confiaba en una cadena de `CHECKPOINT_STATE.json` para `Gate D`, y no extraía el `K` declarado del nombre de archivo/candidata en `Gate F`.
2. **Fallos históricos reales detectados y bloqueados por la suite reparada:**
   - **Gate A (`delta_hits_vs_strong_baseline <= 0`):**
     - `SUBMISSION_CANDIDATE_2_BEST_PURE_LIGHTGBM.csv` (`oof_hits = 65,897`, `delta_hits = -64` vs Podium `65,961`) figuraba con estado `READY_FOR_MANUAL_REVIEW` en `portfolio_tomorrow_manifest.json`. Ahora [`validate_gate_a`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L64-L152) recalcula los hits directamente desde `y_true` y las predicciones OOF y rechaza cualquier candidata con `delta_hits <= 0` o discrepancia entre el manifiesto y los hits reales.
     - `P3_Best_Blend_Dual_Blend` tenía `net_delta_vs_podium = -12` y estado `DEGRADED_IN_HOLDOUT` en [`results/nested_stacking_20261111/parte2_meta_lofo_tournament_results.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/nested_stacking_20261111/parte2_meta_lofo_tournament_results.csv). [`validate_gate_a`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L64-L152) cruza el CSV de evaluación y bloquea la promoción.
   - **Gate B (Fuga de sellers y reutilización de OOF global):**
     - [`scripts/execute_meta_lofo_tournament.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/execute_meta_lofo_tournament.py) usaba `StratifiedKFold(5)` (compartiendo `272` vendedores entre `outer-train` y `outer-val` en el fold 0) y rebanaba los arreglos OOF globales precomputados (`inner_oof_lgb = oof_lgb[trn_idx]`) en lugar de reentrenar por fold. [`validate_gate_b`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L159-L263) inspecciona los índices de partición, los hashes de reentrenamiento y el AST del script, rechazando tanto el script real como cualquier partición con solapamiento de vendedores o early stopping sobre `outer-val`.
   - **Gate C (Probabilidades públicas y `Q05` sin calibración prequential):**
     - Las afirmaciones numéricas de `S5` (`expected_public_score = 0.95057`, `P(>S4) = 98.9%`, `P(>=0.95000) = 66.0%`, `public_q05 = 0.94847`) carecían de artefacto prequential con cobertura empírica medida. [`validate_gate_c`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L270-L353) las rechaza y exige `"NO_IDENTIFIABLE"` salvo que exista un artefacto prequential con $\ge 5$ eventos secuenciales y cobertura empírica recalculada $\ge 0.80$.
   - **Gate D (Cuarentena física de candidatas rechazadas/supersedidas):**
     - Se movieron a [`rejected/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/rejected) un total de **19 archivos CSV** rechazados o supersedidos que permanecían en `submissions/`:
       - `submissions/portfolio_tomorrow_20261111/` $\to$ [`rejected/portfolio_tomorrow_20261111_ag0177_superseded/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/rejected/portfolio_tomorrow_20261111_ag0177_superseded)
       - `submissions/new_bases_portfolio_20261111/` $\to$ [`rejected/new_bases_portfolio_20261111_ag0168/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/rejected/new_bases_portfolio_20261111_ag0168)
       - `submissions/SUBMISSION_5_*` $\to$ [`rejected/s5_broad_frontier_ag0162/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/rejected/s5_broad_frontier_ag0162)
     - [`validate_gate_d`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L360-L437) inspecciona directamente el árbol de directorios `submissions/` y los manifiestos activos, sin confiar en `CHECKPOINT_STATE.json`.
   - **Gate E (Etiquetas metodológicas sin evidencia estructural):**
     - [`validate_gate_e`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L444-L579) bloquea las etiquetas `"nested"`, `"temporal"`, `"leave-family"` y `"bootstrap jerárquico"` (`"seller_category_bootstrap"`) cuando los splits internos filtran índices de `outer_val`, cuando los timestamps violan causalidad ($\max(t_{\text{train}}) \ge \min(t_{\text{val}})$), cuando se filtra un OOF global post-hoc sin reentrenar la familia excluida, o cuando el bootstrap muestrea filas IID dentro de un mismo cluster `(seller, category)`.
   - **Gate F (Discrepancia entre `K15` en el nombre y `8` flips reales):**
     - El archivo [`rejected/portfolio_tomorrow_20261111_ag0177_superseded/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/rejected/portfolio_tomorrow_20261111_ag0177_superseded/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv) declaraba `K15` en su nombre de archivo y de candidata pero sólo tenía `8` flips reales frente a `S4`. [`validate_gate_f`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L586-L664) extrae el `K` declarado mediante expresión regular desde el nombre del archivo y de la candidata y falla inmediatamente (`Declared K=15 != actual flips=8`) aunque el manifiesto declare `8`.
   - **Gate G (Formato de submission):**
     - [`validate_gate_g`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L671-L725) rechaza archivos con IDs duplicados, desordenados, conteo distinto de `30,000`, valores nulos o clases fuera de `{"new", "used"}`.
   - **Gate H (Orden de exposición en reportes):**
     - [`validate_gate_h`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L732-L799) evalúa el archivo real [`coordination/antigravity_to_codex/AG-0177-RESULT.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0177-RESULT.md) y lo rechaza porque su primer encabezado (`## 1. Resumen Ejecutivo del Cierre de Sprint`) no expone primero los resultados negativos ni los gates fallidos, y también rechaza reportes que omitan candidatos degradados como `P3_Best_Blend_Dual_Blend (-12)` en su sección negativa inicial.
3. **Estado de la validación outer seller-group (`T97`):**
   - El experimento completo de 5 outer folds $\times$ 4 inner folds ([`scripts/seller_group_nested_podium_validation.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/seller_group_nested_podium_validation.py)) continúa ejecutándose en segundo plano (`task-14043`, actualmente en el Outer Fold 3 de 5). **Ninguna candidata tiene estado `PROMOTED`, `READY` ni `QUALIFIED`**. No se ha generado ningún CSV de submission ni se ha tocado Kaggle.

---

## 1. Matriz de Cobertura de Gates A–H (`32` Aserciones Red$\to$Green en `8` Gates)

| Gate | Función Validadora | Fixtures Positivos | Fixtures Negativos | Evidencia Estructural Verificada |
| :--- | :--- | :---: | :---: | :--- |
| **Gate A** | [`validate_gate_a`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L64-L152) | 1 | 3 | Recalcula aciertos OOF desde `y_true` + predicciones; exige `delta_hits > 0` vs baseline fuerte (`65,961`). Pasa `+25`; falla con LightGBM `-64` marcado `READY`, con manifiesto falsificado (`66,000` vs `65,897`) y con `P3_Best_Blend_Dual_Blend` (`-12`, `DEGRADED_IN_HOLDOUT`). |
| **Gate B** | [`validate_gate_b`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L159-L263) | 1 | 3 | Verifica disyunción total de `seller_id` en outer e inner folds, `25` hashes `fit_rows_sha` distintos (`5` outer + `20` inner) y AST del script. Falla con `StratifiedKFold` (`272` sellers compartidos), con [`execute_meta_lofo_tournament.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/execute_meta_lofo_tournament.py) rebanando OOF global, y con early stopping en `outer-val`. |
| **Gate C** | [`validate_gate_c`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L270-L353) | 2 | 2 | Pasa cuando los campos públicos declaran `"NO_IDENTIFIABLE"` o cuando existe artefacto prequential con $\ge 5$ eventos secuenciales y cobertura recalculada $\ge 0.80$. Falla con las probabilidades de `S5` (`0.95057`, `98.9%`, `66.0%`, `0.94847`) y con artefacto prequential de cobertura `0.20 < 0.80`. |
| **Gate D** | [`validate_gate_d`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L360-L437) | 1 | 2 | Verifica el árbol de archivos real (`0` rechazados en `submissions/`, `19` CSVs en `rejected/`). Falla si `SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv` permanece en `submissions/` o si un manifiesto activo incluye una entrada `REJECTED_*`. |
| **Gate E** | [`validate_gate_e`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L444-L579) | 4 | 4 | Prueba estructural de los 4 procedimientos: `nested`, `temporal`, `leave-family` y `seller_category_bootstrap`. Falla ante fuga de `outer_val` en inner split, violación causal $\max(t_{\text{train}})=135 \ge \min(t_{\text{val}})=130$, filtrado post-hoc de OOF sin refit y pesos IID dentro de un cluster `(seller, category)`. |
| **Gate F** | [`validate_gate_f`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L586-L664) | 1 | 2 | Recalcula SHA-256, `id_target_hash` unido por `ID`, flips reales (`N2U`+`U2N`) y `K` extraído por regex del nombre de archivo/candidata. Pasa con `S4`; falla con `SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv` (`K=15` vs `8` flips reales) y con `id_target_hash` alterado. |
| **Gate G** | [`validate_gate_g`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L671-L725) | 1 | 2 | Verifica `30,000` filas, columnas `["ID", "condition"]`, IDs `1..30000` únicos y ordenados, clases `{"new", "used"}`, `0` nulos y `used_rate` en `[0.35, 0.60]`. Pasa con `S4` (`0.4722`); falla con IDs duplicados y con valores nulos. |
| **Gate H** | [`validate_gate_h`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py#L732-L799) | 1 | 2 | Verifica que el primer encabezado `## ` exponga resultados negativos/gates fallidos primero, sin insignias promocionales previas y sin omitir fallos conocidos. Pasa con [`AG-0180-ACK-CORRECTION.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0180-ACK-CORRECTION.md); falla con [`AG-0177-RESULT.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0177-RESULT.md) y con reportes que omiten `P3_Best_Blend_Dual_Blend (-12)`. |

---

## 2. Ejecución de la Suite y Artefactos Generados

- **Comando ejecutado:**
  ```powershell
  python -m unittest -v tests/test_preventive_gates.py
  ```
- **Resultado:** `Ran 8 tests in 1.026s — OK` (`32` aserciones de fixtures registradas: `12` positivas + `20` negativas).
- **Artefactos guardados y sus hashes SHA-256:**
  - [`tests/test_preventive_gates.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/tests/test_preventive_gates.py):
    `95be354ff8bd8220e9c05d691d96d77ebcaec087428200310029a08bde49d777`
  - [`results/preventive_gates_repair_20261111/test_suite_stdout.txt`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/preventive_gates_repair_20261111/test_suite_stdout.txt):
    `50d1ecbb5e65f94c6ace55f6f0a5414638f6128c1ce1b38481c6727d8c818a0d`
  - [`results/preventive_gates_repair_20261111/gates_coverage_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/preventive_gates_repair_20261111/gates_coverage_report.json):
    `4928645ca34c2744f74aae5f87988969af418958f2c5518656f49625ffa6cb80`
  - [`results/preventive_gates_repair_20261111/gates_coverage_report.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/preventive_gates_repair_20261111/gates_coverage_report.md)
