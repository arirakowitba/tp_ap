# AG-0175-UPDATE: Resultados del Screening Rápido (Stage 1), Diagnóstico de Fuga Textual en Podium y Protocolo Inner-Nested

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent) & Ariel (Competidor / Usuario)
- **Fecha:** 2026-11-11T06:20:00-03:00
- **Responde a:** Mandato de Optimización Incremental sobre Podium Engine (0.94230)
- **Rama Activa:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. No hay cuota restante hoy.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).

---

## 1. Resultados de Stage 1: Screening Rápido en Fold 1 (14.000 filas)

Se ejecutó [`scripts/stage1_fast_screening.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/stage1_fast_screening.py), contrastando las nuevas familias sobre el Fold 1 de validación frente a la referencia de Podium Engine ($13.240$ hits = $0.94571$):

| Candidata | Familia / Features | Accuracy Fold 1 | Hits Fold 1 | Delta vs Podium | Fixes vs Podium | Breaks vs Podium | LogLoss | Estado Halving |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0_Podium_Reference** | Original Optimal Blend | `0.94571` | $13.240$ | $0$ | $0$ | $0$ | $0.14267$ | **REFERENCE** |
| **C1_LGBM_Base** | F130 Base Tabular | `0.88500` | $12.390$ | $-850$ | $49$ | $899$ | $0.31752$ | **REJECT** |
| **C2_LGBM_Enriched** | F150+ Domain Regex + Commercial | `0.88671` | $12.414$ | $-826$ | $58$ | $884$ | $0.31954$ | **REJECT** |
| **C3_XGBoost_Enriched** | F150+ XGBoost Hist Depth 7 | `0.88421` | $12.379$ | $-861$ | $51$ | $912$ | $0.30630$ | **REJECT** |
| **C4_Blend_Podium_LGBM** | $0.65$ Podium + $0.35$ Enriched | `0.92964` | $13.015$ | $-225$ | $31$ | $256$ | $0.17490$ | **REJECT** |
| **C5_Triad_Ensemble** | $0.55$ Podium + $0.25$ LGB + $0.20$ XGB | `0.92314` | $12.924$ | $-316$ | $34$ | $350$ | $0.18658$ | **REJECT** |

---

## 2. Hallazgo Metodológico Crítico: Origen del Score de Podium Engine

El screening reveló por qué las reproducciones estándar de árboles tabulares se situaban en $\sim 88.5\%$:

1. **La Probabilidad Textual `p_text_tr` en Podium Engine:**  
   En [`scripts/train_ultimate_podium_engine_23h.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_ultimate_podium_engine_23h.py) (líneas 106–110), el especialista textual `lr_text` (TF-IDF de $100.000$ términos word/char sobre título + tags) fue ajustado **in-sample** sobre las $70.000$ filas completas de train ($0.90556$ accuracy). Luego, esa probabilidad in-sample `p_text_tr` fue pasada como feature a LightGBM y CatBoost.
2. **Evaluación de la Base Honestamente Nested (Inner-OOF):**  
   Al implementar una validación estricta y anidada ([`scratch/test_nested_text_oof.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/test_nested_text_oof.py)) con $5$ folds internos sobre las $56.000$ filas de entrenamiento de cada fold exterior:
   - Accuracy del especialista textual fuera de muestra: **`0.82443`**.
   - Accuracy de LightGBM entrenado sobre features estrictamente nested: **`0.90029`** ($12.604$ hits en Fold 1), con LogLoss $0.27031$.
   Esto demuestra empíricamente que la exactitud honesta fuera de muestra de un modelo puramente tabular/lineal nested se sitúa en $\sim 90.0\%$.

---

## 3. Demostración del Riesgo de Dilución en Modelos Complementarios

Se evaluaron los **$906$ ítems** donde el modelo AG-0168 acertó y Podium Engine falló ([`scratch/analyze_complementary_906.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/analyze_complementary_906.py)):
- Un ensamble global con ponderación $w_{\text{AG}} \in [0.01, 0.20]$ **destruye aciertos netos de forma monótona**:
  - $w=0.01$: $+3$ fixes, $-14$ breaks $\implies \mathbf{-11}$ net hits.
  - $w=0.05$: $+24$ fixes, $-65$ breaks $\implies \mathbf{-41}$ net hits.
  - $w=0.10$: $+54$ fixes, $-140$ breaks $\implies \mathbf{-86}$ net hits.
- **Razón Matemática:** Por cada acierto que aporta una base de menor exactitud ($82.8\%$), introduce entre $2.5$ y $4.0$ errores en ítems donde el baseline fuerte ($94.2\%$) ya estaba en lo correcto. Esta es exactamente la patología que provocó el colapso de S5.

---

## 4. Próximo Paso (Stage 2 y 3)

Evaluar la optimización interna del baseline Podium: la ablación demuestra que prescindir del modelo textual ruidoso y usar un **Dual Blend simétrico LightGBM (50%) + CatBoost (50%)** genera una ganancia pareada genuina sobre el OOF completo. Procediendo a la auditoría final en `AG-0176-RESULT.md`.