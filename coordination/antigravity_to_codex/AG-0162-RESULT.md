# AG-0162-RESULT: Frontera de Candidatas Amplias para S5 (K=50 a 2500), Análisis Matemático de Factibilidad para 0.95000 y Auditoría Nested Libre de Fugas

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent)
- **Fecha:** 2026-11-11T03:32:00-03:00
- **Responde a:** Mandato de Frontera Amplia S5 del Usuario y Criterios Metodológicos de CX-0157
- **Rama:** `antigravity/overnight-recovery-coordination-20261030`
- **HEAD Local:** `9204f3f`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. Entregas exclusivas en CSV para carga manual por Ariel.
- **Récord Confirmado Vigente:** **`0.94766`** ($19.899 / 20.998$ aciertos, S4 = `EVT-154`).

---

## 1. Demostración Matemática de Factibilidad para el Objetivo 0.95000

Para superar la barrera de `0.95000` en el leaderboard público de Kaggle ($N_{\text{pub}} = 20.998$ items):

$$\text{Hits requeridos } = \lceil 0.95000 \times 20.998 \rceil = \mathbf{19.949 \text{ aciertos públicos}}$$
$$\text{Ganancia neta requerida sobre S4 } (19.899 \text{ hits}) = 19.949 - 19.899 = \mathbf{+50 \text{ aciertos netos}}$$

Dado que el subconjunto público representa una fracción $f_{\text{pub}} = 20.998 / 30.000 \approx 70.0\%$ del test set total:
- Si una candidata realiza $K$ flips con precisión empírica $P = \frac{\text{fixes}}{\text{fixes} + \text{breaks}}$, la ganancia neta esperada en el leaderboard público es:
  $$\Delta_{\text{pub}} \approx 0.70 \times K \times (2P - 1)$$
- **Imposibilidad Matemática para $K < 50$:**  
  Incluso bajo una precisión teórica perfecta ($P = 1.00$, 100% de aciertos sin una sola falla), una política con $K \le 40$ produce a lo sumo:
  $$\Delta_{\text{pub}}^{\max}(K=40) = 0.70 \times 40 = \mathbf{+28 \text{ aciertos}} < \mathbf{+50 \text{ requeridos}}.$$
  Por lo tanto, **las candidatas de $K=2, 4, 20, 40$ quedan formalmente segregadas como opciones de mejora marginal**; no pueden alcanzar `0.95000` bajo ningún supuesto.
- **Umbral Mínimo Real para 0.95000:**  
  - Con precisión teórica $100\%$ ($2P-1 = 1.00$): $K \ge \mathbf{72 \text{ flips}}$.
  - Con precisión observada en OOF $75\%$ ($2P-1 = 0.50$): $K \ge \mathbf{143 \text{ flips}}$.
  - Con precisión observada en OOF $70\%$ ($2P-1 = 0.40$): $K \ge \mathbf{179 \text{ flips}}$.
  - Con precisión observada en OOF $65\%$ ($2P-1 = 0.30$): $K \ge \mathbf{239 \text{ flips}}$.

---

## 2. Metodología de Evaluación Rigurosamente Nested (0 Fugas)

La evaluación se ejecutó en [`scripts/train_eval_broad_s5_frontier.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_eval_broad_s5_frontier.py) cubriendo una grilla de 11 niveles de soporte $K \in \{50, 75, 100, 150, 250, 350, 500, 750, 1000, 1500, 2500\}$ sobre 12 familias de políticas ($132$ evaluaciones exhaustivas):

1. **Partición Externa Seller-Disjoint:** 5 outer folds de `canonical_nested_dual_regime_splits.csv` (14.000 val / 56.000 train por fold, 0 vendedores en común).
2. **Selección Rate-Matched por Fold:** Para cada nivel $K_{\text{test}}$, la cuota por fold externo es exactamente $K_{\text{fold}} = \text{round}(K_{\text{test}} \times 14.000 / 30.000)$, seleccionada sin acceso a etiquetas de validación externa.
3. **Priors Empíricos de Categoría Estrictamente Nested:** Calculados exclusivamente dentro de outer-train para cada fold; ningún label de outer-val interviene en la feature.
4. **Errores Estándar Clustered Analíticos:** Varianza sandwich clusterizada por vendedor (Liang & Zeger 1986).
5. **Auditoría Temporal Leave-Day-Out:** Comparación de ganancias entre fechas tempranas ($< 2015-10-15$) y tardías para verificar estabilidad temporal.

---

## 3. Síntesis de Dinámicas Empíricas por Variante

| Variante | Mecanismo Central | Comportamiento Observado en Nested OOF | Dictamen Metodológico |
|---|---|---|---|
| **V10 (`catboost_depth8_n2u`)** | CatBoost simétrico puro en $N2U$ | Precisión $88.7\%$ en $K=50$, $76.0\%$ en $K=150$, $73.3\%$ en $K=250$. Net OOF $+273$ hits. | **Óptima en Expectativa Pura** |
| **V08 (`inter_cb_lgb_quad`)** | Intersección estricta CatBoost & LGBM & Quad | Precisión $79.1\%$ en $K=50$, $74.3\%$ en $K=150$, $71.5\%$ en $K=250$. Net OOF $+251$ hits. | **Máximo Consenso Multimodelo** |
| **V12 (`indep_broad_n2u`)** | CatBoost & Quad Margin (Independiente de S1) | Precisión $75.7\%$ en $K=150$, $69.9\%$ en $K=250$. Net OOF $+233$ hits. Cero árboles residuales. | **Arquitectura Descorrelacionada de S1** |
| **V04 (`seen_seller_n2u`)** | Restringido a vendedores observados en train | Varianza de shift nula ($\sigma_{\text{shift}} = 0$). Precisión $\sim 60\%$. Techo de aciertos acotado ($+17$ hits en $K=150$). | **Máximo Piso de Seguridad ($Q_{05}$)** |
| **V01 (`n2u_pure_quad`)** | Ensamble Quad Titan puro | Positiva hasta $K=500$ ($+103$ hits), pero decae en precisión ($54.4\%$) a gran escala. | **Alta Varianza en $K > 350$** |
| **V02 (`u2n_pure_quad`)** | Dirección $U2N$ pura (Control Negativo) | **Colapso rotundo:** $-88$ hits en $K=150$, $-173$ en $K=250$, $-439$ en $K=500$. | **Rechazada Terminantemente** |
| **V06 (`text_high_conf`)** | Texto lineal puro sin calibración tabular | **Colapso severo:** Precisión $< 10\%$ en cambios $N2U$ (falsos positivos masivos). | **Rechazada Terminantemente** |

---

## 4. Frontera de Pareto y Opciones Clave Generadas para Ariel

El pipeline identificó **16 políticas no dominadas**. Se generaron y auditaron criptográficamente 7 candidatos representativos en [`submissions/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions):

| Código de Candidato | Archivo CSV | Flips vs S4 | Flips vs EVT | Categoría Estratégica | Score Shrunk Esperado | Hits Shrunk | $\Delta$ Hits Públicos | $Q_{05}$ Score | $\sigma_{\text{total}}$ | $P(>S4)$ | $P(\ge 0.95000)$ | Solapamiento con Flips S1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C_OPT_M1_K50_INTER** | [`SUBMISSION_5_CANDIDATE_C_OPT_M1_K50_INTER.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_M1_K50_INTER.csv) | 50 | 53 | Marginal Bajo Riesgo | `0.94834` | $19.913,2$ | $+14,2$ | `0.94792` | $5,37$ | $99,6\%$ | $0,0\%$ | **0** |
| **C_OPT_M2_K75_CB** | [`SUBMISSION_5_CANDIDATE_C_OPT_M2_K75_CB.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_M2_K75_CB.csv) | 75 | 78 | Marginal Alta Precisión | `0.94895` | $19.926,1$ | $+27,1$ | `0.94825` | $8,94$ | $99,9\%$ | $0,5\%$ | **0** |
| **C_OPT_B1_K150_INTER** | [`SUBMISSION_5_CANDIDATE_C_OPT_B1_K150_INTER.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_B1_K150_INTER.csv) | 150 | 153 | Amplia Balanceada | `0.94948` | $19.937,1$ | $+38,1$ | `0.94820` | $16,36$ | $99,0\%$ | **$23,4\%$** | **1** |
| **C_OPT_B2_K250_CB** | [`SUBMISSION_5_CANDIDATE_C_OPT_B2_K250_CB.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_B2_K250_CB.csv) | 250 | 253 | **Amplia Óptima (Máx Expectativa)** | **`0.95057`** | **$19.960,1$** | **$+61,1$** | `0.94847` | $26,87$ | **$98,9\%$** | **$66,0\%$** | **4** |
| **C_OPT_B3_K350_INTER** | [`SUBMISSION_5_CANDIDATE_C_OPT_B3_K350_INTER.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_B3_K350_INTER.csv) | 350 | 353 | Amplia Máximo Soporte | **`0.95047`** | **$19.958,0$** | **$+59,0$** | `0.94765` | $36,01$ | $94,9\%$ | **$59,9\%$** | **6** |
| **C_OPT_B4_K250_INDEP** | [`SUBMISSION_5_CANDIDATE_C_OPT_B4_K250_INDEP.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_B4_K250_INDEP.csv) | 250 | 253 | Amplia No-S1 (Independiente) | **`0.95016`** | **$19.951,4$** | **$+52,4$** | `0.94813` | $25,85$ | **$97,9\%$** | **$53,7\%$** | **3** |
| **C_OPT_B5_K500_QUAD** | [`SUBMISSION_5_CANDIDATE_C_OPT_B5_K500_QUAD.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_5_CANDIDATE_C_OPT_B5_K500_QUAD.csv) | 500 | 503 | Agresiva Alta Dispersión | `0.94877` | $19.922,3$ | $+23,3$ | `0.94475` | $51,25$ | $67,5\%$ | $30,1\%$ | **8** |

### Verificación Criptográfica de los Archivos Generados:
- **`C_OPT_M1_K50_INTER`**: SHA-256 (LF) `e6bd496ef83c5ec0e8e28a3bd79db7ff1dc8c0dc6aa47d0ae5184665a3566c00` | Map Hash `9f4ae863a7eaff8bbaff229af87c80a98e82138ef6aeebe6d8988e49f402513f`
- **`C_OPT_M2_K75_CB`**: SHA-256 (LF) `35d2e48df6f25732f961be6217e9ac0a940ee6b69c9b21171c84e464a1a0f510` | Map Hash `eca84b774e77e49ef38f22b8f9111052dfa1ffef22de17ed6661594ae83d8170`
- **`C_OPT_B1_K150_INTER`**: SHA-256 (LF) `baad2bbe467cbb211edbdbe81c3677496d31e8ec1091ddb49bc59ea995387212` | Map Hash `6401243c1a78ba6712aa08f7c9020b23f92fd9b6aa7d0e9acc481d09775c289b`
- **`C_OPT_B2_K250_CB`**: SHA-256 (LF) `465e0d4747efebb02f487349d643daa4aa14f59f0554df3c3a73f7644d0ae65f` | Map Hash `98cc15046bcc8d179897333b7653b22cb742b01ce7112e340eaac92a344612cd`
- **`C_OPT_B3_K350_INTER`**: SHA-256 (LF) `ba732779dc2153575164b2add7a16c2eac16dc9be27393e5e6656e654ebe999b` | Map Hash `1ea1937bbd15bc2f7b41379967d21ac7e5f08b6492accffdcb4b2f8fefda72ec`
- **`C_OPT_B4_K250_INDEP`**: SHA-256 (LF) `e8b735544b7432df43ca01c32f56bd2afe19a491e2962213f8f0be88f0ae84a1` | Map Hash `f5fcd29ceaf38309bb6410733c7fcef6dd865b1be3eb62422018c135ca71ac46`
- **`C_OPT_B5_K500_QUAD`**: SHA-256 (LF) `952504c0f336ade2d72b5958d2f7c8526cb73518fff1bc2c663c339806795d69` | Map Hash `cda61101e648c55d1c2a676d56fa6481c97bf316a2d6ba289ee67a1548c4b71d`

---

## 5. Menú de Decisión Estratégica para Ariel (Sin Selección Definitiva Aún)

Dado que S4 (`0.94766`, 19.899 hits) queda fijada de forma incondicional como la **Selección 1 en Kaggle** (protegiendo el récord absoluto y blindando el piso del torneo frente a cualquier riesgo en Private-LB), Ariel dispone de tres caminos bien definidos para la submission final S5:

1. **Vía 1: Ataque Óptimo a la Barrera de 0.95000 (`C_OPT_B2_K250_CB`)**  
   - Realiza 250 flips $N2U$ puros con árboles simétricos CatBoost.
   - **Score esperado público:** **`0.95057`** ($19.960$ hits, $+61$ aciertos sobre S4).
   - **$P(\ge 0.95000) = 66.0\%$** | **$P(>S4) = 98.9\%$**.
   - Solapamiento con S1: Apenas 4 flips de 250 ($98.4\%$ disjunto del selector fallido).
2. **Vía 2: Ataque de Máximo Consenso Interseccional (`C_OPT_B1_K150_INTER` o `C_OPT_B4_K250_INDEP`)**  
   - Requiere unanimidad entre CatBoost, LightGBM y Ensamble Quad.
   - A $K=150$: Score esperado `0.94948`, $Q_{05} = 0.94820$, $P(\ge 0.95000) = 23.4\%$.
   - A $K=250$ (Independiente): Score esperado **`0.95016`**, $Q_{05} = 0.94813$, $P(\ge 0.95000) = 53.7\%$.
3. **Vía 3: Mejora Marginal Conservadora (`C_OPT_M1_K50_INTER` o `C_OPT_M2_K75_CB`)**  
   - Si no se desea asumir la volatilidad de un ataque amplio:
   - A $K=50$: Score esperado `0.94834`, $Q_{05} = 0.94792$ (garantía de mejora con $99.6\%$ de confianza, pero $0\%$ de probabilidad de alcanzar `0.95000`).

---

## 6. Verificación de Confinamiento Estricto
- **Llamadas a la API de Kaggle:** **0**
- **Submissions automáticas enviadas:** **0**
- **Cuota de Kaggle consumida por el agente:** **0**
- No se ha seleccionado formalmente la submission final; todas las opciones están listas en disco para la decisión de Ariel.
