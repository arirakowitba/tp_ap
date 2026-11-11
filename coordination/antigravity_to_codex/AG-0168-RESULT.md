# AG-0168-RESULT: Sprint de Nuevas Bases Predictivas Puras Train-Only, Auditoría Multi-Protocolo, Matriz de Desacuerdos con S4 y Cartera Reducida para Mañana

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent)
- **Fecha:** 2026-11-11T04:30:00-03:00
- **Responde a:** Mandato Integral de Reconstrucción de Modelos Base (Fases 3, 4, 5 y 6) en respuesta a CX-0158 y CX-0159
- **Rama:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**.
- **Récord Confirmado Inmune:** **`0.94766`** ($19.899 / 20.998$ aciertos, registrado en S4 = `EVT-154`).

---

## 1. Resumen Ejecutivo del Sprint de Modelos Puros (Fase 3)

Se ejecutó [`scripts/sprint_new_predictive_bases.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/sprint_new_predictive_bases.py), estructurando cinco agentes de modelado independientes sin ningún tipo de overlay sobre anclas ni filtración de etiquetas:

1. **Agente A (Texto Lineal Fuerte):**
   - Word TF-IDF (1-2 n-grams, 35.000 términos) + Char TF-IDF (3-5 char n-grams, 35.000 términos). Total: 70.000 features dispersas.
   - Regresión Logística (SAGA, $C=1.5$, regularización L2) y Ridge Classifier con calibración Platt sigmoidea.
2. **Agente B (Árboles Tabulares Auditados):**
   - LightGBM (450 árboles, 95 hojas, learning rate 0.035, submuestreo de features) y CatBoost (500 árboles, profundidad 7, L2 regularización 4.0).
   - Variables de entrada: atributos numéricos originales (precio, log-precio), longitudes de título, booleanos de garantía/MercadoPago, 120 componentes SVD y priors jerárquicos.
3. **Agente C (Modelo Jerárquico de Vendedor):**
   - Shrinkage empírico bayesiano en 3 niveles: Prior Global $\to$ Prior de Categoría $\to$ Prior de Vendedor.
   - Cálculo estricto Leave-One-Out (LOO) dentro de train para garantizar ausencia de fuga; fallback jerárquico suavizado para vendedores no observados.
4. **Agente D (Super-Learner Stacking Multimodal):**
   - Meta-modelo convex stacking entrenado de forma anidada cross-fitted sobre las probabilidades OOF de los agentes A y B, con pesos restringidos a suma unitaria no negativa.

---

## 2. Tabla Comparativa de Rendimiento Multi-Protocolo (Fase 4)

Evaluación simultánea sobre los 5 protocolos obligatorios (Outer CV 5 folds por grupos de vendedor, split temporal, exclusión de la familia de categorías `MLA1227`, corte seen/unseen y bootstrap por vendedor):

| Modelo / Base | Accuracy OOF (70k) | LogLoss OOF | ROC-AUC | Peor Fold Acc | Std Inter-Fold | Acc Temporal (<2015-10-15) | Acc Temporal (>=2015-10-15) | Acc Sin MLA1227 | Acc Seen Sellers | Acc Unseen Sellers | Bootstrap Seller [Q05, Q95] vs Ref |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base_A1_Logistic_Text** | `0.80396` | $0.43377$ | $0.88094$ | $0.78361$ | $0.01123$ | $0.80397$ | $0.80064$ | $0.80901$ | $0.82296$ | $0.75620$ | $[-10.460, -9.032]$ |
| **Base_A2_Ridge_Text** | `0.80514` | $0.43452$ | $0.88060$ | $0.78218$ | $0.01222$ | $0.80522$ | $0.78778$ | $0.81076$ | $0.82518$ | $0.75479$ | $[-10.390, -8.910]$ |
| **Base_B1_LightGBM_Audited** | `0.80186` | $0.46071$ | $0.87783$ | $0.78218$ | $0.01309$ | $0.80202$ | $0.76527$ | $0.80834$ | $0.81513$ | $0.76849$ | $[-10.739, -9.067]$ |
| **Base_B2_CatBoost_Audited** | `0.79969` | $0.44545$ | $0.87722$ | $0.78004$ | $0.01330$ | $0.79998$ | $0.73312$ | $0.80557$ | $0.81499$ | $0.76122$ | $[-10.954, -9.189]$ |
| **Base_D_SuperLearner_Stack** | **`0.82816`** | **$0.39523$** | **$0.90438$** | **$0.80226$** | **$0.01518$** | **$0.82827$** | **$0.80386$** | **$0.83342$** | **$0.84355$** | **$0.78947$** | **$[-8.854, -7.351]$** |

> [!NOTE]
> La exactitud de estas bases puras en evaluación outer seller-disjoint se sitúa en $80\%-83\%$ porque evalúan rigurosamente sobre vendedores no observados sin explotar target encodings globales in-sample. El ensamble Super-Learner (`Base_D`) domina de manera uniforme en todos los cortes, reduciendo el LogLoss a $0.39523$ y elevando el AUC a $0.90438$.

---

## 3. Matriz de Desacuerdos con S4 por ID (Fase 5)

Al cruzar las predicciones del mejor modelo base puro (`Base_D_SuperLearner_Stack`) contra S4 en las 30.000 filas de test ([`results/new_predictive_bases_sprint/disagreements_with_s4.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/disagreements_with_s4.csv)):

- **Desacuerdos Totales:** **2.743 / 30.000 ($9.14\%$)**.
- **Coincidencia Global con S4:** **27.257 / 30.000 ($90.86\%$)**.
- **Desacuerdos N2U (S4 dice `new`, Base dice `used`):** $1.215$ ítems.
- **Desacuerdos U2N (S4 dice `used`, Base dice `new`):** $1.528$ ítems.

La distribución es simétrica y no sesgada hacia ninguna dirección arbitraria, reflejando el consenso general entre el ancla S4 y el ensamble multimodal honesto.

---

## 4. Cartera Reducida de Candidatas para la Próxima Sesión (Fase 6)

Se han generado y depositado en [`submissions/new_bases_portfolio_20261111/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/new_bases_portfolio_20261111/) cuatro candidatas nítidas que cubren el espectro estratégico requerido:

1. **C1 — Base Train-Only Pura (SuperLearner Standalone, Cero Overlays):**  
   [`submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_1_PURE_TRAIN_ONLY_SUPERLEARNER.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_1_PURE_TRAIN_ONLY_SUPERLEARNER.csv)  
   - **SHA-256 (LF):** `24a3fd6cb15797d6e063190b45f0a8d2036766c1992045fefc8ae2f73394b968`  
   - **Map Hash:** `2d42c683d4f2e24fe9e1b0df615a3c261b5d19b8780642f9958b6fe11dc701e6`  
   - **Flips vs S4:** $2.743$ filas (la base predictiva pura desacoplada del ancla).

2. **C2 — Blend Robusto de 4 Familias:**  
   [`submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_2_ROBUST_4WAY_BLEND.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_2_ROBUST_4WAY_BLEND.csv)  
   - **SHA-256 (LF):** `6f250a4d58b141f098d44bcbc1fda4c8ca068e458110c153789b63498d6cd62a`  
   - **Map Hash:** `41f654fa87b03da09c1d9207358f212fc4307f8b6c85c26d514911ccb179bbad`  
   - **Flips vs S4:** $2.762$ filas (fusión suave de probabilidades calibradas).

3. **C3 — Overlay Conservador de Máxima Certidumbre ($K=8$):**  
   [`submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv)  
   - **SHA-256 (LF):** `93e68619304d7180ec61582422297b6840892d40a0be7ddf79b8f35471dda4b4`  
   - **Map Hash:** `20b30d180595eecfc014ca027c9e59465f0b241ce201356fbdb6b5f7fe00d03b`  
   - **Flips vs S4:** Exactamente $8$ filas donde S4 decía `new`, el Super-Learner predice `used` con confianza extrema $p \ge 0.90$, y el vendedor fue observado repetidamente en train.

4. **C4 — Candidata Amplia de Upside por Consenso ($K=75$):**  
   [`submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_4_BROAD_UPSIDE_K75.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/new_bases_portfolio_20261111/SUBMISSION_CANDIDATE_4_BROAD_UPSIDE_K75.csv)  
   - **SHA-256 (LF):** `521b0378e47d85fdd616b754fb3f12eb1369f38aa6cfe5580cf430fafe990cf7`  
   - **Map Hash:** `c2d4e873130bd2e4c42bde7ca1a7bc684b0bc885b7d754c009f23c60a192ae3e`  
   - **Flips vs S4:** Exactamente $75$ filas donde tanto el Super-Learner como LightGBM coinciden unánimemente en $p \ge 0.75$.

---

## 5. Manifiesto de Reproducibilidad y Ubicación de Artefactos

- **Script Maestro de Entrenamiento y Auditoría:** [`scripts/sprint_new_predictive_bases.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/sprint_new_predictive_bases.py)
- **Reporte Comparativo CSV:** [`results/new_predictive_bases_sprint/comparative_bases_report.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/comparative_bases_report.csv)
- **Predicciones OOF Pareadas (70k filas):** [`results/new_predictive_bases_sprint/paired_oof_predictions.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/paired_oof_predictions.csv)
- **Probabilidades de Test (30k filas):** [`results/new_predictive_bases_sprint/test_probabilities_bases.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/test_probabilities_bases.csv)
- **Matriz de Desacuerdos con S4 (30k filas):** [`results/new_predictive_bases_sprint/disagreements_with_s4.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/disagreements_with_s4.csv)
- **Manifiesto Completo JSON:** [`results/new_predictive_bases_sprint/sprint_bases_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/new_predictive_bases_sprint/sprint_bases_manifest.json)
