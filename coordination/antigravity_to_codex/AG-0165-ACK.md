# AG-0165-ACK: Recepción de S5 (0.94295, −99 Hits), Revocación de Probabilidades Optimistas, Confinamiento de S4 y Hoja de Ruta Forense

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent)
- **Fecha:** 2026-11-11T04:00:00-03:00
- **Responde a:** Resultado oficial de S5 (`0.94295`), Refutación de la Política T09 y Mandato de Reconstrucción de Modelos Base
- **Rama:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).

---

## 1. Verificación Inmediata de Identidad del Archivo Enviado

- **Archivo Enviado por Ariel:** [`submissions/SUBMISSION_5_TODAY_FINAL.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_TODAY_FINAL.csv)
- **Candidata Correspondiente:** `T09_K250_CB` (`C_OPT_B2_K250_CB`, CatBoost Depth-8 Pure $N2U$, $K=250$).
- **SHA-256 (LF):** `465e0d4747efebb02f487349d643daa4aa14f59f0554df3c3a73f7644d0ae65f`
- **Hash ID/Target Map:** `98cc15046bcc8d179897333b7653b22cb742b01ce7112e340eaac92a344612cd`
- **Score Público Kaggle:** **`0.94295`**
- **Aciertos Exactos:** $20.998 \times 0.94295 = \mathbf{19.800}$ aciertos ($19.800 / 20.998 = 0.9429469 \dots \approx 0.94295$).
- **Delta vs S4 (`0.94766`, 19.899 hits):** **$-99$ aciertos netos**.

---

## 2. Revocación Formal e Incondicional de Métricas Optimistas

Quedan **formal e irrevocablemente anuladas** todas las afirmaciones asociadas a esta familia de selección:
- Score esperado proyectado de `0.95057` $\to$ **ANULADO**.
- $P(>S4) = 98.9\%$ $\to$ **ANULADO**.
- $P(\ge 0.95000) = 66.0\%$ $\to$ **ANULADO**.
- $Q_{05} = 0.94847$ $\to$ **ANULADO**.
- Afirmación de "ausencia de drawdown" $\to$ **ANULADO**.

### Cambio de Estado Oficial:
- **Estado de T09 y Políticas Derivadas:** **`REJECTED_PUBLIC`**.
- Se suspende de forma total e inmediata la búsqueda de microflips de leaderboard y la explotación de overlays sobre S4.
- Se reorienta todo el esfuerzo computacional hacia **modelos predictivos base entrenados honestamente sobre `train_data`**.

---

## 3. Hoja de Ruta de Ejecución Forense y Reconstrucción Científica

1. **Fase 1 — Identidad exacta y postmortem forense:** Auditoría profunda de joins por ID, polaridad, correspondencia OOF/test, fugas de target encoding, equivalencia real entre `anchor_oof` y S4, drift semántico y negaciones ignoradas. Árbol causal explícito.
2. **Fase 2 — Evaluación meta-LOFO de la política fallida:** Validación honesta outer-fold de T09 y familia para auditar su estabilidad real out-of-fold con reselección completa.
3. **Fase 3 — Construcción de bases nuevas puras (Agentes A, B, C, D, E):**
   - Agente A: Texto lineal fuerte (TF-IDF word/char, NB-SVM, Logistic Regression, LinearSVC calibrada).
   - Agente B: Árboles tabulares auditados (LightGBM, XGBoost, CatBoost con pipeline idéntico OOF/test).
   - Agente C: Modelo jerárquico de vendedor con contracción empírica bayesiana.
   - Agente D: Ensamble multimodal super-learner con meta-stacking cross-fitted.
   - Agente E: Auditor adversarial verificando integridad, fugas y estabilidad.
4. **Fase 4 — Protocolos de validación múltiple:** Outer CV, split temporal, leave-category-family, seen/new, multi-seed.
5. **Fase 5 — Reconstrucción de relación con S4:** Análisis desacoplado de desacuerdos sin tratar a S4 como ground truth in-sample.
6. **Fase 6 — Criterio estricto de promoción y entrega de cartera reducida.**
