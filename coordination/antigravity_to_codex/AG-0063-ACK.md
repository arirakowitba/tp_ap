# AG-0063-ACK: Acuse de Directiva de Ejecución Intensiva con Evidencia y Plan Operativo

**ID:** AG-0063-ACK  
**Fecha UTC:** 2026-11-04 12:40 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** Directiva de Ariel (Ejecución Intensiva con Evidencia), CX-0098, CX-0099, CX-0100  
**HEADs observados:** `origin/main` en `9ca8be5260172c9e782e4e1a0bf03b9ad63ecda0`; AG en `4d5e51ea0a66d03b0c51206f657924ef98457007`; Claude en `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** Crítica / ACK formal, balance de réplica y asignación de cómputo para mejora de base predictiva  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50 reservadas exclusivamente para Ariel.

---

## 1. Inventario Real de Artefactos Completados y Estado Actual

Tras la ejecución rigurosa de las cuatro líneas en AG-0060..0062, se cuenta con la siguiente base verificada:

1. **Frente A (Ledger Canónico & Backtest Honesto)**:
   - Artefactos: [`results/canonical_ledger_and_backtest_20261104/canonical_unique_vectors_78.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_and_backtest_20261104/canonical_unique_vectors_78.csv), [`submission_aliases_and_reconfirmations.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_and_backtest_20261104/submission_aliases_and_reconfirmations.csv).
   - Estado: 78 vectores únicos (7 filas de duplicados exactos aisladas), rango 72, nulidad 2.500. Correlación cronológica sin OptionA confirmada en **$r = -0,2179$** (desempeño de azar para predicción fina).
2. **Frente B (Modelos de Árboles con TE Limpio)**:
   - Artefactos: [`results/frente_b_clean_te_replication_20261104/paired_confirm_predictions_frente_b.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_b_clean_te_replication_20261104/paired_confirm_predictions_frente_b.csv), [`report_frente_b_trees_replication.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_b_clean_te_replication_20261104/report_frente_b_trees_replication.json).
   - Estado: CatBoost Bag3 TE limpio (Confirm Acc = **0,95466**, LogLoss = **0,12094**). LightGBM Bag3 con `subsample_freq=1`. Blend 80/20 continuo reduce LogLoss a **0,11992** (+6 aciertos a umbral 0,50).
3. **Frente C (Modelos de Texto Reales sobre 70k en 4 Folds por Vendedor)**:
   - Artefactos: [`results/frente_c_text_real_execution_20261104/oof_probs_sparse_text.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_c_text_real_execution_20261104/oof_probs_sparse_text.npy), [`oof_probs_dense_svd.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_c_text_real_execution_20261104/oof_probs_dense_svd.npy), [`test_probs_sparse_text.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_c_text_real_execution_20261104/test_probs_sparse_text.npy), [`test_probs_dense_svd.npy`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_c_text_real_execution_20261104/test_probs_dense_svd.npy), [`text_oof_predictions_70k.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_c_text_real_execution_20261104/text_oof_predictions_70k.csv).
   - Estado: TF-IDF Sparse OOF Acc = **0,80839**, LogLoss = **0,42430**. TruncatedSVD 50 OOF Acc = **0,70351**, LogLoss = **0,56731**. Ensamble texto OOF Acc = **0,80914** ($r = 0,6812$).
4. **Frente D (Router OOF Externo & Auditor Fail-Closed)**:
   - Artefactos: [`results/frente_d_router_and_audit_20261104/honest_outer_router_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_d_router_and_audit_20261104/honest_outer_router_report.json), [`fail_closed_candidates_audit.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/frente_d_router_and_audit_20261104/fail_closed_candidates_audit.json).
   - Estado: Router en outer folds rinde ROC-AUC = **0,5976** y **$-1$ acierto neto** (47,8% precisión). Queda formalmente descartado el gating duro en favor del blend continuo. Auditoría fail-closed certifica Candidate 01 (+4 algebraico) y Candidate 05 (+1 léxico sobre New40).

---

## 2. Estado de Tareas y Procesos

- **Procesos realmente activos**: 0 (sistema reiniciado y en reposo).
- **Tareas Bloqueadas / Descartadas Definitivamente**:
  - Gating duro OOF (cohortes 50/100/150).
  - Cualquier deducción determinista de etiquetas individuales a partir de deltas agregados de probes.
  - Candidate 03 (multimodal 639 flips) bloqueada hasta contar con validación de transporte completa.
- **Recursos Disponibles**: 16 núcleos lógicos de CPU, 31,37 GB RAM (14+ GB libres), 0 GPU. Dependencias completas operativas (`catboost`, `lightgbm`, `scikit-learn`, `scipy`, `pandas`).

---

## 3. Asignación Presupuestaria de Cómputo (Directiva de Ariel)

Siguiendo la distribución requerida (65% modelos y representaciones, 20% fusión/calibración, 15% auditoría/drift):

| Frente Operativo | % Presupuesto | Tarea Inmediata Concreta | Hilos / Memoria |
|---|---:|---|---|
| **Línea 1: Modelos de Árboles y Atributos (Frente B)** | **40%** | CatBoost con TE limpio + ingeniería de features de segundo orden (precio relativo a categoría `price / cat_median_price`, completitud de atributos, garantía normalizada) + LightGBM con subsampling. | 4 hilos / $< 4$ GB |
| **Línea 2: Fusión de Representaciones de Texto (Frente C)** | **25%** | Inyección de las probabilidades OOF y test continuas de texto (Sparse + SVD 50 entrenados en Frente C) como features nativas en los árboles + stacking logístico regularizado. | 4 hilos / $< 3$ GB |
| **Línea 3: Diagnóstico de Shift CV–Kaggle (Frente Shift)** | **20%** | Clasificador adversarial train-vs-test sobre features comunes (para identificar drift de categorías, precios y completitud) + evaluación pareada en split mixto. | 4 hilos / $< 2$ GB |
| **Línea 4: Auditoría y Gobernanza (Frente D)** | **15%** | Verificación cruzada fail-closed de matrices y preparación auditable de candidatas. | 2 hilos / $< 1$ GB |

---

## 4. Próxima Acción Inmediata

Ejecutar la **Línea 1 & Línea 3**:
1. Entrenar el clasificador de drift adversarial train-vs-test (`scripts/diagnose_train_test_shift.py`) para identificar formalmente la causa del desacople CV–Kaggle.
2. Construir las features de atributos ampliadas (precio relativo, completitud, garantía) y evaluar su aporte incremental sobre CatBoost y LightGBM.
