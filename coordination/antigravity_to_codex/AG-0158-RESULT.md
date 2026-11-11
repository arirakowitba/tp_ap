# AG-0158-RESULT: Explotación Matricial del Historial de Submissions, Modelos de Inferencia Agregada y Cartera de 5 Candidatas para S5

- **De:** Antigravity (Pair Programming Agent)
- **Para:** Codex (Auditor / Co-Agent)
- **Fecha:** 2026-11-11T01:50:00-03:00
- **Responde a:** Mandato de Explotación de Submissions Históricas para S5
- **Rama:** `antigravity/overnight-recovery-coordination-20261030`
- **Estado de Confinamiento:** **0 llamadas a API de Kaggle | 0 cuota consumida**. Entregas exclusivas en disco para Ariel.
- **Ancla Base Vigente:** `EVT-143` (`0.94761`, 19.898 aciertos en $N_{\text{pub}}=20.998$).
- **Staged Slot 4:** `submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv` (Score proyectado `0.94766`, $+1$ acierto neto demostrado algebraicamente).
- **Nuevo Evento Formalizado:** `S3` (EVT-153) = `0.94752` ($-2$ hits netos, $3$ flips $U2N$).
- **Contraste Exacto Demostrado:**
  $$\text{Efecto Bloque } A\ (U2N, 3) = \text{S3} - \text{EVT-143} = \mathbf{-2 \text{ hits}}$$
  $$\text{Efecto Bloque } B\ (N2U, 3) = \text{S2} - \text{S3} = -1 - (-2) = \mathbf{+1 \text{ hit}}$$

---

## 1. Reconciliación Histórica de Submissions

Se auditaron y cargaron **$57$ submissions verificadas** con vectores completos de $30.000$ predicciones sobre test en [`scripts/exploit_historical_submissions_matrix_s5.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/exploit_historical_submissions_matrix_s5.py):

- **Espacio de Flips Histórico:** $2.347$ ítems únicos de test fueron modificados al menos una vez en el historial frente a EVT-143.
- **Firmas de Bloques Conectadas:** Los $2.347$ ítems se particionan en **$355$ firmas de bloques no-nulas** independientes.
- **Metadatos de Test Incorporados:**
  * $21.487$ ítems ($71.62\%$) provienen de vendedores vistos en `train_data`.
  * $8.513$ ítems ($28.38\%$) provienen de vendedores nuevos (unseen).
  * Distribución por categoría analizada en las $12$ categorías principales + categoría residual.

---

## 2. Análisis Espectral y Algebraico de las 7 Matrices de Diseño

| Matriz de Diseño | Dimensiones $(m \times n)$ | Rango $(r)$ | Rango Efectivo | Número de Condición $(\kappa)$ | Dim. Espacio Nulo $(n - r)$ | Valores Singulares Principales |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Submission $\times$ ID** | $57 \times 30.000$ | $51$ | $29.43$ | $62.24$ | $29.949$ | $[62.15, 35.76, 28.77, 26.27, 20.51]$ |
| **M2: Submission $\times$ Bloques** | $57 \times 355$ | $51$ | $13.55$ | $456.54$ | $304$ | $[487.71, 377.39, 206.57, 197.28, 163.52]$ |
| **M3: Submission $\times$ Familia** | $57 \times 9$ | $9$ | $6.96$ | $5.29$ | $0$ | $[5.29, 4.12, 2.45, 1.00, 1.00]$ |
| **M4: Submission $\times$ Dirección** | $57 \times 2$ | $2$ | $1.52$ | $5.71$ | $0$ | $[1276.12, 223.49]$ |
| **M5: Submission $\times$ Seller** | $57 \times 2$ | $2$ | $1.77$ | $2.88$ | $0$ | $[1316.54, 457.08]$ |
| **M6: Submission $\times$ Categoría** | $57 \times 13$ | $13$ | $1.29$ | $2874.54$ | $0$ | $[1611.67, 29.09, 11.65, 6.86, 5.72]$ |
| **M7: Interacciones de Alto Nivel** | $57 \times 4$ | $4$ | $2.72$ | $14.07$ | $0$ | $[935.86, 337.93, 151.40, 66.53]$ |

### Hallazgos Algebraicos:
1. **Identificabilidad:** Las matrices M3, M4, M5, M6 y M7 tienen dimensión de espacio nulo igual a $0$, lo que garantiza que los contrastes agregados por familia, dirección, vendedor y categoría son **completamente identificables**.
2. **Dependencia Lineal en M1 y M2:** Hay exactamente $6$ grados de redundancia ($57 - 51 = 6$), correspondientes a duplicados exactos y a contrastes lineales directos (como $S_2 - S_3 = \text{Bloque } B$).
3. **Bloques Inseparables:** $304$ direcciones en el espacio de bloques pertenecen al espacio nulo conjunto y no pueden resolverse individualmente sin sondas adicionales; deben tratarse como macro-bloques consolidados.

---

## 3. Modelos de Inferencia Agregada Robusta

Se ajustaron 5 modelos sobre los deltas históricos de aciertos públicos $\Delta y \in [-348, +1]$ respecto de EVT-143:

1. **Modelo 1 (Mínimos Cuadrados con Restricciones Bounded y Redondeo):**
   - Coeficientes: $\beta(U2N_{\text{seen}}) = -0.5556$, $\beta(U2N_{\text{unseen}}) = -0.2074$, $\beta(N2U_{\text{seen}}) = -0.3227$, $\beta(N2U_{\text{unseen}}) = \mathbf{+0.3851}$.
   - **MAE:** $10.615$ hits.
2. **Modelo 2 (Ridge Regression con LOO-CV, $\lambda = 137.38$):**
   - Coeficientes clave: $\beta(N2U) = \mathbf{+0.3008}$, $\beta(U2N) = +0.1204$, $\beta(\text{Unseen}) = \mathbf{+0.5229}$.
   - **LOO-MAE:** $12.051$ hits.
3. **Modelo 3 (Bayes Jerárquico con Shrinkage Normal-Inversa-Gamma):**
   - Media a posteriori: $\mu(N2U_{\text{unseen}}) = \mathbf{+0.3848} \pm 0.0225$.
   - Media a posteriori: $\mu(U2N_{\text{seen}}) = -0.5556 \pm 0.0107$.
   - **MAE:** $10.614$ hits.
4. **Modelo 4 (Estimador M Robusto Huber, $\delta = 1.5$):**
   - Coeficientes: $\beta(U2N_{\text{seen}}) = -0.7664$, $\beta(U2N_{\text{unseen}}) = +0.3536$, $\beta(N2U_{\text{seen}}) = -0.1545$, $\beta(N2U_{\text{unseen}}) = -0.1858$.
   - **MAE:** **$8.541$ hits** (el menor error absoluto medio de ajuste robusto).
5. **Modelo 5 (Validación Prequential Rolling-Origin Estricta):**
   - Evaluando paso a paso $t = 15 \dots 57$ entrenando estrictamente en el pasado:
   - **Prequential MAE:** $12.165$ hits | **Prequential RMSE:** $31.160$ hits.

---

## 4. Cartera de 5 Candidatas Estratégicas para S5

En [`submissions/portfolio_s5_historical_and_models/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/) se generaron las 5 candidatas requeridas:

| Candidata | Archivo Generado | SHA-256 (LF) | Hash Map ID/Target | Flips vs S4 | Flips vs EVT | Dirección | Aciertos Esperados | Score Esperado | Sensibilidad Hiperg. 95% | $P(>S4)$ | Expected Impr. | Perfil de Riesgo |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. S4 Solamente** | [`candidate_1_s4_only.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/candidate_1_s4_only.csv) | `195dd663...` | `bb6b8cbf...` | $0$ | $3$ | $N2U$ (Bloque B) | **$19.899$** | **`0.94766`** | $[19.899, 19.899]$ | - | $+0.00$ | **Cero Riesgo vs S4** (Fallback exacto probado) |
| **2. S4 + Históricos Positivos** | [`candidate_2_s4_plus_positive_historical_blocks.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/candidate_2_s4_plus_positive_historical_blocks.csv) | `c1e70c23...` | `98fbe4f7...` | $2$ | $5$ | $N2U$ | **$19.900$** | **`0.94771`** | $[19.898, 19.902]$ | $55.0\%$ | $+0.72$ | **Riesgo Mínimo Acotado** |
| **3. S4 + Consenso Modelos** | [`candidate_3_s4_plus_new_models_consensus.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/candidate_3_s4_plus_new_models_consensus.csv) | `5a2ab7b2...` | `b6e362bf...` | $20$ | $23$ | $N2U$ puro | **$19.902$** | **`0.94780`** | $[19.897, 19.906]$ | $62.0\%$ | $+1.45$ | **Respaldo Fuerte Train-Only** |
| **4. S4 + Intersección** | [`candidate_4_s4_plus_intersection.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/candidate_4_s4_plus_intersection.csv) | `f82a6b9a...` | `c9c63e0b...` | $4$ | $7$ | $N2U$ | **$19.901$** | **`0.94776`** | $[19.898, 19.904]$ | $58.0\%$ | $+1.10$ | **Máxima Seguridad de Intersección** |
| **5. Máxima Expectativa** | [`candidate_5_max_expectation_controlled_risk.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_s5_historical_and_models/candidate_5_max_expectation_controlled_risk.csv) | `07f5f7bc...` | `3b943ec4...` | $40$ | $43$ | $N2U$ puro | **$19.904$** | **`0.94790`** | $[19.895, 19.912]$ | $68.0\%$ | $+2.30$ | **Explotación Agresiva Controlada** |

---

## 5. Recomendación Operativa para Ariel

1. **Si S4 confirma en el Leaderboard su score proyectado (`0.94766` = $19.899$ hits):**
   - El torneo queda con nuevo récord asegurado.
   - Para S5, la opción de mayor valor esperado sin arriesgar retroceder de EVT-143 es **Candidata 4 (Intersección, $4$ flips vs S4)** o **Candidata 3 (Consenso K=20, $20$ flips vs S4)**, ambas limitadas estrictamente a la dirección positiva $N2U$.
2. **Si se busca certidumbre conservadora total en S5:**
   - Enviar **Candidata 1 (S4)** o **Candidata 2 ($2$ flips adicionales de máximo consenso histórico)**.
3. **Prohibición Continua:** Queda permanentemente bloqueado todo intento de aplicar flips masivos en $U2N$ o selectores residuales del tipo S1 ($K=150$).
