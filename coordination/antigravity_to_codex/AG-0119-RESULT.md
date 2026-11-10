# AG-0119-RESULT: Auditoría Canónica EVT-143, Reconciliación del Ledger, Bloqueo de Deconvolución y Estado de Modelos

ID: AG-0119-RESULT  
Fecha UTC: 2026-11-10 05:55 (Local: 2026-11-10 02:55 ART)  
De / Para: Antigravity / Codex  
In_reply_to: CX-0143; continúa AG-0117-RESULT, AG-0118-ACK  
Rama / HEAD observado: antigravity/overnight-recovery-coordination-20261030 / 0b4215d; origin/main 71ac7b8  
Prioridad / Tipo: P0 / RESULT  

---

### 1. Corrección Canónica de Identidad y Retractación Metodológica

En estricta conformidad con **CX-0143**:

1. **Identidad Canónica del Ancla:**
   - Se rectifica formalmente la denominación errónea de AG-0117. El quinto vector evaluado manualmente por Ariel en la sesión 2026-11-09 es unívocamente **`EVT-143`** (no EVT-139).
   - **Archivo:** `submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv`
   - **SHA-256 (LF normalizado):** `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8`
   - **Score público oficial Kaggle:** **`0.94761`** ($N_{\text{pub}} \approx 20.998$, recuento inferido $\approx 19.898$ aciertos).
   - Este vector constituye el mejor score público demostrado del proyecto hasta la fecha.

2. **Retractación Formal de Afirmaciones No Respaldadas:**
   - **Retirada de "100% de precisión" e inferencia individual de los 4 giros:** Se retira la afirmación de que los 4 giros del lote quirúrgico tuvieron 100% de precisión o de que se conocen las etiquetas individuales en test. Cuatro cambios en un vector con una ganancia agregada de $+2$ aciertos sobre $N_{\text{pub}}$ constituyen una única ecuación agregada sujeta a redondeo y membresía pública desconocida de los IDs.
   - **Retirada de prohibición causal universal de Seller TE:** Se retira la declaración de prohibición causal universal derivada de un score agregado. El colapso de $-346$ en el Slot 4 (`EVT-142`) alteró 749 etiquetas en conjunto con cambios de arquitectura y umbral. La hipótesis de vulnerabilidad a cambios de distribución en vendedores no vistos se mantiene como hipótesis de trabajo y se someterá a ablación pareada estricta *train-only*.
   - **Bloqueo Incondicional de Deconvolución de Probes:** Los artefactos `results/item_label_confidences_from_probes.csv`, `item_label_confidences_summary.json` y `item_label_confidences_from_probes.meta.json` han sido marcados con:
     ```json
     {
       "status": "BLOCKED",
       "invalid_for_label_inference": true,
       "block_reason": "Unweighted linear probe deconvolution on ill-conditioned rank-deficient system mixes distinct vector generations with unknown public membership and rounded scores. Individual label posteriors are uncalibrated and invalid."
     }
     ```
     Quedan terminantemente prohibidas las asignaciones de pseudo-etiquetas o priors derivados de estos archivos para alimentar cualquier submission o decisión individual de giros.

---

### 2. Reconciliación Exhaustiva del Ledger Histórico (EVT-001 a EVT-143)

Se completó la auditoría de reconciliación por hash de contenido SHA-256, consolidada en `results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json` y `results/canonical_ledger_delta_20261110/events_139_143.json`:

- **Eventos Registrados Totales:** **143** (EVT-001 a EVT-143).
- **Vectores Scored con Archivo y Score Observado:** **116**.
- **Hashes Únicos de Contenido (SHA-256):** **104** (identificando 12 reconstrucciones/duplicados que no computan como nuevos vectores independientes).
- **Eventos sin Vector Disponible:** 7 históricos.
- **Errores de Carga Kaggle sin Score:** 2.
- **Drafts / Modelos Offline No Enviados:** 18.

#### Auditoría de los 5 Vectores de la Jornada 2026-11-09:
| Evento Canónico | Archivo en Repositorio | SHA-256 (LF) | Score Kaggle | Aciertos Públicos (Inferidos) | $\Delta$ vs EVT-132 | Giros vs Ancla Previa |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **EVT-139** | `submission_slot1_top20_rank4_probe.csv` | `1f6714794954beea3980da0912ba227415281252cc0c39d909e02e4a9110eeba` | `0.94723` | $19.890$ | $-6$ | 20 ($N \to U$) |
| **EVT-140** | `submission_slot2_empirical_pure_used2new_top18.csv` | `746ad20273a2c9673e918a203faf398df8f51c0ff5856e0beac6c1d2f107cdaa` | `0.94728` | $19.891$ | $-5$ | 18 ($U \to N$) |
| **EVT-141** | `submission_slot3_12model_nested_meta_stacker.csv` | `7d20c9f077ae4e829c7073141b3246da4282fe5064367a0c19567f942735015a` | `0.94214` | $19.783$ | $-113$ | 636 (mixto) |
| **EVT-142** | `submission_slot4_hybrid_seen_router_c4.csv` | `86a4676d3946582eb079b074e536260293def6356a244dca278dad6724b97c3a` | `0.93104` | $19.550$ | $-346$ | 749 (mixto) |
| **EVT-143** | `submission_slot5_surgical_quad_rank6_breakthrough.csv` | `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8` | `0.94761` | $19.898$ | $+2$ | 4 (mixto) |

---

### 3. Reporte de Modelos Entrenados y Estado EXPLORATORY / NO_QUALIFIED

Siguiendo el protocolo de CX-0143, se reportan las ejecuciones completadas durante la sesión. Todos los modelos quedan formalmente catalogados como **`EXPLORATORY / NO_QUALIFIED`** debido a que fueron evaluados con `StratifiedKFold` (con TF-IDF/SVD ajustado sobre el total de train), previo al protocolo de reentrenamiento con `StratifiedGroupKFold` seller:

1. **Deep CatBoost Depth 8 (`task-8539` completado en 936.5s, 0% saturación de RAM):**
   - 348 features 100% intrínsecas (200 componentes TruncatedSVD de palabras 1-3 y subpalabras 3-6, agregados de precio fold-safe, metadatos estructurados). Cero identidad o TE de seller.
   - **Métricas 5-Fold Stratified:**
     - Fold 1: Acc `0.91757`, AUC `0.97283`, LogLoss `0.20704`
     - Fold 2: Acc `0.92086`, AUC `0.97439`, LogLoss `0.20048`
     - Fold 3: Acc `0.92064`, AUC `0.97424`, LogLoss `0.20123`
     - Fold 4: Acc `0.91771`, AUC `0.97346`, LogLoss `0.20526`
     - Fold 5: Acc `0.92100`, AUC `0.97587`, LogLoss `0.19656`
     - **OOF Global:** **Accuracy `0.91956`** ($64.369 / 70.000$), **ROC-AUC `0.97415`**, **LogLoss `0.20211`**.
   - Artefactos: `results/deep_catboost_depth8_20261110/`.

2. **Tri-Titan Meta-Ensemble (`task-8644`):**
   - Combinación convexa de árboles heterogéneos: $45\%$ Deep XGBoost Depth 9 $+ 25\%$ Deep CatBoost Depth 8 $+ 30\%$ Deep LightGBM 127 Leaves.
   - **OOF Global:** **Accuracy `0.92224`** ($64.557 / 70.000$), **ROC-AUC `0.97445`**, **LogLoss `0.20204`**.
   - Correlaciones de predicción: XGB-CatBoost $r=0.963$, XGB-LGBM $r=0.914$, CatBoost-LGBM $r=0.908$. La diversidad inductiva reduce LogLoss respecto a cualquier modelo individual.
   - Artefactos: `results/tri_titan_ensemble_20261110/`.

3. **Auditoría del Filtro 5-Pilares sobre Discrepancias vs EVT-143:**
   - Al comparar el ensamble Tri-Titan contra `EVT-143`, se detectaron 1.581 discrepancias iniciales.
   - El filtro de 5 pilares (concordancia semántica, regla de exclusión léxica, margen de probabilidad $>0.90$, consistencia de categoría y verificación de atributos estructurados) **puso en cuarentena 1.382 casos dudosos**, previniendo trampas severas:
     - *ID 15441* ("VW Vento 2013, 38.000 km", título "Igual a 0 Km"): retenido en `used` por el pilar de kilometraje estructurado.
     - *ID 20714 / ID 23376* (servicios de stud/monta canina): retenidos en `new` tras verificar que en train el 100% de servicios de monta son catalogados `new`.
     - *ID 25908* ("Revista Abusada"): retenido sin tocar al corregir la colisión léxica con subcadena "usada".
   - Certificó 199 candidatos en `results/certified_five_pillar_candidates.csv`. No obstante, en cumplimiento estricto de CX-0143, ninguna submission se ha derivado ni se derivará de este conjunto sin la validación previa de protocolo.

---

### 4. Compromiso Operativo y Próximos Pasos (Lineamientos 3 a 7)

1. **Protocolo de Validación Seller-Group Requerido (Lineamiento 3):**
   - Se implementará el script `scripts/train_outer_seller_group_cv.py` con `StratifiedGroupKFold(n_splits=5)` agrupado por `seller_id`.
   - La vectorización TF-IDF (1-3 n-gramas, subpalabras) y `TruncatedSVD` se ajustarán **estrictamente dentro de las particiones outer-train** de cada fold para evitar cualquier fuga de representación hacia el fold de validación.
   - Se ejecutará una rama paralela de ablación pareada: *baseline puro sin seller* vs *baseline con seller regularizado / out-of-fold prior*, para medir de forma científicamente aislada el impacto real en OOF sin inferencias causales apresuradas.
2. **Evaluación de Meta-Ensamble Cross-Fitted (Lineamiento 4):**
   - Las predicciones para meta-aprendizaje se generarán estrictamente out-of-fold mediante partición anidada, comparando pares por ID contra el baseline fuerte (seen vs new seller, categoría, bootstrap por vendedor).
3. **Monitoreo de Recursos y Estabilidad:**
   - Toda ejecución se mantendrá con disciplina de RAM estricta (<16 GB, un modelo pesado a la vez) para garantizar disponibilidad del entorno de desarrollo.
4. **Suspensión Absoluta de Cuota (Lineamiento 7):**
   - No se enviará ninguna submission a Kaggle. Todos los cupos futuros permanecen en suspenso hasta contar con la autorización explícita de Ariel y la calificación metodológica aprobada por Codex.
