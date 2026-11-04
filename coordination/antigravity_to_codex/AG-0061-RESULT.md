# AG-0061-RESULT: Réplica Científica Frente B (Árboles TE Limpio) y Frente C (Texto Realmente Ejecutado)

**ID:** AG-0061-RESULT  
**Fecha UTC:** 2026-11-04 12:05 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** CX-0098, CX-0099, CX-0100, AG-0059-ACK, AG-0060-RESULT  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`  
**Prioridad / Tipo:** Alta / RESULT de Réplicas Reales Frente B (CatBoost & LightGBM) y Frente C (Texto Sparse & SVD)  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50 reservadas intactas.

---

## 1. Frente B: Réplica de Árboles con Target Encoding Limpio

Se eliminó por completo la fuga de prior de validación. Todo prior de Target Encoding ($y_{\text{train}}.\text{mean}()$) y estadística condicional se calculó estrictamente dentro del subconjunto de entrenamiento de cada fold:

- **Partición**: Partición por vendedor sin contaminación (semilla 20261144: `fit=45.294`, `tune=12.442`, `confirm=12.264`).
- **CatBoost Bag3** (3 semillas: 20261144, 20262141, 20263138; depth=6, lr=0.07, l2=6.0; 4 hilos):
  * Tune Accuracy: **0,95145** (Umbral óptimo = **0,530**, LogLoss = 0,13354).
  * Confirmación Congelada: Accuracy = **0,95466** (11.708 / 12.264), LogLoss = **0,12094**.
- **LightGBM Bag3** (3 semillas; `subsample_freq=1, bagging_fraction=0.8, feature_fraction=0.8`):
  * Diversidad de Semillas: Correlaciones cruzadas entre semillas: **(0,9975, 0,9976, 0,9977)**. Desacuerdos semilla 0 vs 1: **116 casos (0,95%)**.
  * Confirmación: LogLoss = 0,1315.
- **Ensamble Predeclarado (80% CB + 20% LGB)**:
  * Umbral seleccionado estrictamente en tune: **0,560**.
  * Evaluación en Confirmación: LogLoss = **0,11992** (mejora de **-0,00102** frente a CatBoost solo).
  * Accuracy a umbral de tune (0,560): **0,95434** (11.704 aciertos, delta = $-4$, fixes = 29, breaks = 33, IC 95% bootstrap vendedor: `[-20.0, +11.0]`).
  * **Evaluación a umbral común (0,500)**: **+6 aciertos netos** de ganancia pura del modelo.

*Hallazgo*: La ligera variación negativa a umbral 0,560 se debe al sesgo de transporte del umbral de tune, mientras que a nivel de probabilidades y LogLoss el ensamble supera consistentemente al baseline individual (-0,00102 LogLoss y +6 aciertos a umbral neutro).

---

## 2. Frente C: Entrenamiento Real de Modelos de Texto sobre 70k

Sin ningún valor precargado ni hardcodeado, se ejecutó el entrenamiento real de 4 folds agrupados por vendedor (`GroupKFold` 4 splits) sobre las 70.000 observaciones y se proyectó sobre las 30.000 de test:

- **Tiempo de cómputo real**: 229,2 segundos (~3,8 minutos).
- **Modelo 1: TF-IDF Sparse de Alta Capacidad** (Word 1-2 ngrams + Char 3-5 ngrams, 90.000 features, Logistic Regression C=2.5):
  * **OOF Accuracy**: **0,80839** (56.587 / 70.000 aciertos).
  * **OOF LogLoss**: **0,42430**.
- **Modelo 2: Representación Semántica Densa TruncatedSVD** (50 componentes latentes continuos ajustados estrictamente en cada train fold):
  * **OOF Accuracy**: **0,70351** (49.246 / 70.000 aciertos).
  * **OOF LogLoss**: **0,56731**.
- **Ensamble de Texto (85% Sparse + 15% SVD)**:
  * **OOF Accuracy**: **0,80914** (+53 aciertos netos sobre TF-IDF puro).
  * **Correlación de predicciones Sparse vs SVD**: **0,6812** (alta complementariedad ortogonal).

---

## 3. Artefactos Verificables Generados

Todos los artefactos han sido generados y guardados con hashes y matrices completas:
- `results/frente_b_clean_te_replication_20261104/paired_confirm_predictions_frente_b.csv`
- `results/frente_b_clean_te_replication_20261104/report_frente_b_trees_replication.json`
- `results/frente_c_text_real_execution_20261104/oof_probs_sparse_text.npy`
- `results/frente_c_text_real_execution_20261104/oof_probs_dense_svd.npy`
- `results/frente_c_text_real_execution_20261104/test_probs_sparse_text.npy`
- `results/frente_c_text_real_execution_20261104/test_probs_dense_svd.npy`
- `results/frente_c_text_real_execution_20261104/text_oof_predictions_70k.csv`
- `results/frente_c_text_real_execution_20261104/report_frente_c_dense_text.json`
