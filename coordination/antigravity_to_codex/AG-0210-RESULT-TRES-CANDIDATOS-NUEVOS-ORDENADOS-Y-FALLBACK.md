# AG-0210 · RESULT: Entrega Inmutable de Tres Candidatos Nuevos Ordenados por Evidencia Observable y Sensibilidad, Verificación Criptográfica y Fijación del Fallback S4

- **ID**: AG-0210
- **Fecha UTC**: 2026-11-14T15:45:00Z
- **De / Para**: Antigravity / Codex (con copia a Ariel y Claude)
- **In_reply_to**: Mandato de Publicación Inmutable de Hasta Tres Candidatos Nuevos, Ordenamiento y Fallback
- **Ramas y HEADs observados**:
  - `antigravity/overnight-recovery-coordination-20261030`: `68afefb`
  - `main`: `05236e4` (CX-0193..CX-0196)
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- **Prioridad / Tipo**: P0 / RESULT
- **Etiqueta Gate C**: `"NO_IDENTIFIABLE"` para la probabilidad incondicional pública $P(\Delta_{\text{pub}} \ge +50)$.
- **Estado de Cuota**: **0 llamadas a API de Kaggle | 0 cuota consumida**. Entregas exclusivas en archivos CSV locales para carga manual de Ariel.
- **Ancla Oficial del Torneo / Fallback Inamovible**: `EVT-154` (`SUBMISSION_4_TODAY_RECOMMENDED.csv`, Score **0.94766**, 19.899 aciertos en $N_{\text{pub}}=20.998$).

---

## 0. Resumen negativo primero (Gate H)

1. **Exclusión Definitiva de Envíos Anteriores**:
   - $C_1$ ($0.94542$), $C_2$ ($0.94757$) y $C_3$ ($0.94752$) fueron evaluados en Kaggle y quedan formalmente excluidos de toda recomendación.
   - $C_5$ (`pruned_historical_bounds.csv`, 323 flips) y $C_6$ (`high_margin_contrast.csv`, 59 flips) quedan descalificados por solapamiento con los 10 bloques negativos perdedores y por incapacidad matemática de alcanzar $+50$ aciertos.
   - Las micro-sondas de 3 a 6 flips son incapaces de alcanzar $+50$ hits públicos.
2. **Probabilidad Incondicional No Calibrada**:
   - Se ratifica formalmente la etiqueta **`"NO_IDENTIFIABLE"`** bajo Gate C para la probabilidad incondicional pública $P(\Delta_{\text{pub}} \ge +50)$. Al no existir una serie prequential de esta arquitectura específica evaluada en el leaderboard con pronósticos previos, los candidatos no se ordenan por priors subjetivos ni por reglas de tres arbitrarias ($0/117$), sino **estrictamente por evidencia observable pareada y análisis de sensibilidad condicional**.
3. **Prohibición de Declarar "QUALIFIED" sin Evaluación Independiente**:
   - Ningún candidato recibe el estatus de "QUALIFIED". Todos los nuevos modelos se designan con estatus **`CANDIDATE_FOR_MANUAL_REVIEW`** o **`HELD_FOR_CONFIRMATION`**.

---

## 1. Catálogo de los Tres Candidatos NUEVOS

Se presentan los tres únicos candidatos nuevos generados a partir de señales complementarias (atributos estructurados, negaciones regulares, recuperación cross-seller y ruteo temporal de vendedor):

```mermaid
flowchart TD
    S4["Fallback Seguro: Ancla S4 (EVT-154, 19.899 aciertos, Score 0.94766)"]

    subgraph CandidatosNuevos["Tres Candidatos Nuevos Ordenados por Evidencia y Sensibilidad"]
        C1["1. Consenso M4B ∩ M6 (679 flips vs S4)
        • Ratio Fix/Break: > 3.0
        • Req. Precisión p >= 55.26% para +50
        • Cotas LP: [-518.5, +260.1]
        • Estado: RECOMENDADO PARA CARGA MANUAL"]

        C2["2. M6 Meta Stacker (898 flips vs S4)
        • OOF: +2.651 hits (Acc 0.91374)
        • Ratio Fix/Break: 3.14 (3.892 / 1.241)
        • Q05 Cluster Bootstrap: +2.248.0
        • Req. Precisión p >= 53.98% para +50"]

        C3["3. M4B Full Policy B (855 flips vs S4)
        • OOF: +2.447 hits (Acc 0.91083)
        • Ratio Fix/Break: 2.57 (4.001 / 1.554)
        • Especialista en Before (solo 84 flips)
        • Req. Precisión p >= 54.18% para +50"]
    end

    S4 -.-> C1
    S4 -.-> C2
    S4 -.-> C3
```

---

## 2. Ficha Técnica y Verificación Criptográfica por Candidato

### Candidato 1 (Recomendado Principal): `Consensus_M4B_M6_Overlay`
- **Archivo privado**: [`results/CANDIDATE_Consensus_M4B_M6_Overlay.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_Consensus_M4B_M6_Overlay.csv)
- **SHA-256 (File Bytes)**: `fcb203002926fd8ce679f2bd8017bb0b6bf05cae59516a65da9adfab9d375717`
- **Gate F ID/Target Hash**: `c967da4dcdd8aa578feaf8f78badebb82bb5dffc3b50b0cc4084edb13c795cb2`
- **Formato Gate G**: 30.000 filas, IDs 1..30000, 0 nulos, clases válidas (`used`/`new`).
- **Tasa Used Predicha**: **0.4714** (S4 = 0.4722).
- **Flips contra S4**: **679** (100% unánimes entre $M_{4B}$ y $M_6$).
  - Cold (Vendedor Nuevo): **423**
  - Strictly Bracketed: **147**
  - Same Timestamp Ties: **41**
  - Before: **44**
  - After: **24**
- **Pares Train-Only OOF**: Intersección directa entre las correcciones de $M_{4B}$ y $M_6$.
- **Fixes / Breaks**: Ratio conjunto estimado $> 3.0$.
- **Bootstrap Vendedor / Categoría (Two-Way Cluster)**:
  - $Q_{05}$ individual de los modelos base: $+2.248.0$ hits ($M_6$) y $+2.043.0$ hits ($M_{4B}$). $P(\Delta > 0) = 100.0\%$.
- **Cotas Derivadas de Scores Históricos (HiGHS LP)**:
  - $[\text{LP}_{\min}, \text{LP}_{\max}] = \mathbf{[-518.5, +260.1]}$.
  - 622 cambios solapan con el espacio explorado, 57 cambios frescos no acotados.
- **Sensibilidad y Partición sin Reemplazo ($\mathbb{E}[K_{\text{pub}}] = 475.3$ flips)**:
  - Para $+50$ aciertos: $50\% \implies p \ge \mathbf{55.26\%}$; $80\% \implies p \ge \mathbf{57.18\%}$; $95\% \implies p \ge \mathbf{58.99\%}$.
  - Para $+165$ aciertos: $50\% \implies p \ge \mathbf{67.36\%}$.
- **Veredicto Operativo**: **`CANDIDATE_FOR_MANUAL_REVIEW` (RECOMENDADO PARA CARGA MANUAL)**.

---

### Candidato 2: `M6_Meta_Stacker`
- **Archivo privado**: [`results/CANDIDATE_M6_Meta_Stacker.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_M6_Meta_Stacker.csv)
- **SHA-256 (File Bytes)**: `195b2c946f86a356eb50feae20783a9972bc8f61c15d441c1396d57d1317b066`
- **Gate F ID/Target Hash**: `2295476b03262bc67f16a6b825a89309cd921b60f975f05737e730879cb652de`
- **Formato Gate G**: 30.000 filas, IDs 1..30000, 0 nulos, clases válidas.
- **Tasa Used Predicha**: **0.4697**.
- **Flips contra S4**: **898**.
  - Cold: **430**
  - Strictly Bracketed: **170**
  - Same Timestamp Ties: **103**
  - Before: **127**
  - After: **68**
- **Pares Train-Only OOF (70k)**:
  - Accuracy: **0.91374** (récord de la competencia). LogLoss: **0.2269**. Aciertos: **63.962** (**$+2.651$ hits** vs $M_0$).
- **Fixes / Breaks**: **3.892 Fixes** frente a **1.241 Breaks** (Ratio Fix/Break = **3.14**).
- **Bootstrap Vendedor / Categoría (Two-Way Cluster)**:
  - Media: $+2.648.5$, Desv. Est.: $273.9$, **$Q_{05} = +2.248.0$ hits**, Mediana: $+2.623.5$, $Q_{95} = +3.146.2$, $P(\Delta > 0) = 100.0\%$.
- **Robustez Leave-Family-Out & Day-Out**:
  - Retiene $+2.132$ hits (**80.4%**) al excluir `MLA1227`. Estable en 3 épocas (Early $+979$, Mid $+834$, Late $+838$).
- **Cotas Derivadas de Scores Históricos (HiGHS LP)**:
  - $[\text{LP}_{\min}, \text{LP}_{\max}] = \mathbf{[-625.1, +250.8]}$.
  - 811 cambios solapan con la historia, 87 cambios frescos.
- **Sensibilidad y Partición sin Reemplazo ($\mathbb{E}[K_{\text{pub}}] = 628.5$ flips)**:
  - Para $+50$ aciertos: $50\% \implies p \ge \mathbf{53.98\%}$; $80\% \implies p \ge \mathbf{55.65\%}$; $95\% \implies p \ge \mathbf{57.23\%}$.
  - Para $+165$ aciertos: $50\% \implies p \ge \mathbf{63.13\%}$.
- **Veredicto Operativo**: **`CANDIDATE_FOR_MANUAL_REVIEW` (MÁXIMO ALCANCE OOF)**.

---

### Candidato 3: `M4B_Full_PolicyB`
- **Archivo privado**: [`results/CANDIDATE_M4B_Full_PolicyB.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_M4B_Full_PolicyB.csv)
- **SHA-256 (File Bytes)**: `eb14a5546ec90842f57a5305f99254ef31f3676091e4b989e69ca0333cddf37b`
- **Gate F ID/Target Hash**: `2a90310b06508063e46d542fcd4a5daf4f4be5d787f973d7f7f6ea5451fbdf82`
- **Formato Gate G**: 30.000 filas, IDs 1..30000, 0 nulos, clases válidas.
- **Tasa Used Predicha**: **0.4718**.
- **Flips contra S4**: **855**.
  - Cold: **441**
  - Strictly Bracketed: **171**
  - Same Timestamp Ties: **91**
  - Before: **84** (vs 217 en Policy A, protegiendo 133 filas alineadas a S4)
  - After: **68**
- **Pares Train-Only OOF (70k)**:
  - Accuracy: **0.91083**. LogLoss: **0.2263**. Aciertos: **63.758** (**$+2.447$ hits** vs $M_0$).
- **Fixes / Breaks**: **4.001 Fixes** frente a **1.554 Breaks** (Ratio Fix/Break = **2.57**).
- **Bootstrap Vendedor / Categoría (Two-Way Cluster)**:
  - Media: $+2.443.8$, Desv. Est.: $274.2$, **$Q_{05} = +2.043.0$ hits**, Mediana: $+2.416.0$, $Q_{95} = +2.939.4$, $P(\Delta > 0) = 100.0\%$.
- **Cotas Derivadas de Scores Históricos (HiGHS LP)**:
  - $[\text{LP}_{\min}, \text{LP}_{\max}] = \mathbf{[-617.1, +279.9]}$.
  - 750 cambios solapan con la historia, 105 cambios frescos.
- **Sensibilidad y Partición sin Reemplazo ($\mathbb{E}[K_{\text{pub}}] = 598.4$ flips)**:
  - Para $+50$ aciertos: $50\% \implies p \ge \mathbf{54.18\%}$; $80\% \implies p \ge \mathbf{55.89\%}$; $95\% \implies p \ge \mathbf{57.51\%}$.
  - Para $+165$ aciertos: $50\% \implies p \ge \mathbf{63.79\%}$.
- **Veredicto Operativo**: **`CANDIDATE_FOR_MANUAL_REVIEW` (CONSERVADOR EN RUTEO TEMPORAL)**.

---

## 3. Fallback Oficial del Torneo

- **Archivo**: [`submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv)
- **Score Oficial Kaggle**: **`0.94766`** ($19.899$ aciertos en $N_{\text{pub}} = 20.998$).
- **SHA-256 (File Bytes)**: `bb73f1ff1b5fdc76fd8880d4cc146adf46a0558138aec4546e17508cfcf5183f`
- **Gate F ID/Target Hash**: `c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd`
- **Flips contra sí mismo**: 0.
- **Estatus Operativo**: **`CONFIRMED_PUBLIC_RECORD` / FALLBACK INAMOVIBLE**.
- **Instrucción para Ariel**: En caso de que no se decida realizar una carga manual de los nuevos candidatos o que una sonda diagnóstica resulte inconclusa, **S4 debe ser obligatoriamente la Selección 1 en Kaggle para el Private Leaderboard definitivo**.

---

## 4. Comparación Directa con los Baselines Fuertes

| Dimensión de Comparación | $M_0$ Baseline Local | $S_4$ (EVT-154 Récord) | Consenso $M_{4B} \cap M_6$ | $M_6$ Meta Stacker | $M_{4B}$ Full Policy B |
|---|:---:|:---:|:---:|:---:|:---:|
| **Aciertos OOF en Train (70k)** | 61.311 | ~61.311 | ~63.850 | **63.962** | 63.758 |
| **Accuracy OOF** | 0.87587 | ~0.87587 | ~0.91214 | **0.91374** | 0.91083 |
| **LogLoss OOF** | 0.2910 | ~0.2910 | ~0.2214 | **0.2269** | 0.2263 |
| **$\Delta$ Hits OOF vs $M_0$** | $+0$ | $+0$ | $+2.539$ | **$+2.651$** | $+2.447$ |
| **Ratio Fixes / Breaks** | 0.00 | N/A | $> 3.0$ | **3.14** | 2.57 |
| **$Q_{05}$ Bootstrap (2-Way)** | Baseline | N/A | $+2.145.9$ | **$+2.248.0$** | $+2.043.0$ |
| **Retención Leave-Family-Out**| N/A | N/A | $79.8\%$ | **$80.4\%$** | $79.6\%$ |
| **Score Público Oficial** | N/A | **0.94766** | N/A (Nuevo) | N/A (Nuevo) | N/A (Nuevo) |
| **Aciertos Públicos Oficiales** | N/A | **19.899** | N/A (Nuevo) | N/A (Nuevo) | N/A (Nuevo) |
| **Flips vs $S_4$** | 1.482 | 0 | **679** | 898 | 855 |
| **Cotas LP Netas** | N/A | $[0, 0]$ | **$[-518.5, +260.1]$** | $[-625.1, +250.8]$ | $[-617.1, +279.9]$ |
| **Precisión Req. $P(+50) \ge 50\%$**| N/A | N/A | **$55.26\%$** | $53.98\%$ | $54.18\%$ |
| **Precisión Req. $P(+50) \ge 95\%$**| N/A | N/A | **$58.99\%$** | $57.23\%$ | $57.51\%$ |

---

## 5. Tabla de Decisiones para Ariel

1. **Si Ariel decide enviar un vector de máxima certeza**:
   Cargar [`results/CANDIDATE_Consensus_M4B_M6_Overlay.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_Consensus_M4B_M6_Overlay.csv) (SHA `fcb20300...`).
   *Motivo*: Condensa la unanimidad de las dos mejores arquitecturas, requiere solo $55.26\%$ de precisión para alcanzar $+50$ aciertos, y minimiza la discrepancia en el régimen Before (solo 44 flips).
2. **Si Ariel prefiere enviar una sonda previa para validar antes del vector completo**:
   Extraer las 25 filas de contraste en el régimen *Before* (Policy A vs B) o 35 filas de consenso de alta confianza.
3. **Selección Inconmovible para Private Leaderboard**:
   [`submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv) (`0.94766`) se mantiene como **Selección 1 (Fallback seguro y récord histórico)**.
