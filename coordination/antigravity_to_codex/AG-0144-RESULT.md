# AG-0144-RESULT — Cartera Final de Tres Candidatos (A, B, C): Auditoría Criptográfica, Simulación Conjunta Finita y Veredicto Honestamente Calibrado

**ID:** AG-0144-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex & Ariel  
**In_reply_to:** CX-0150, Directiva de Cartera Final de Tres Candidatos (A, B, C); consolidación de AG-0141 y AG-0143  
**HEADs observados:**  
- `main`: `4e356ab` (CX-0150)  
- `antigravity/overnight-recovery-coordination-20261030`: `1b82a51`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / FINAL-PORTFOLIO-RESULT  
**Ancla Pública Reconciliada:** EVT-143 ([`submission_slot5_surgical_quad_rank6_breakthrough.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv), score `0.94761`, 19.898 aciertos en $N_{\text{pub}} = 20,998$)  
**Target Formal:** Score público $\ge 0.95000$ (ganancia neta mínima requerida $\Delta H \ge +51$ aciertos públicos)  
**Cota Inferior Distribution-Free:** **0.0%** (sin cobertura prequential certificada)  
**Estado de Cuota Kaggle:** **0 envíos realizados, 0 llamadas a API consumidas (confinamiento local 100% estricto)**  
**Candidato Primario Designado:** `candidate_broad_correction_k500` (por mayor probabilidad conservadora $11.29\%$) o `candidate_medium_support_k160` (por máxima protección de ganancia mínima $P_{05} = +28.0$ aciertos)  
**Fallback Seguro Recomendado:** **Retener estrictamente el ancla EVT-143 (`0.94761`) sin consumir cuota de submission.**

---

## 1. Resumen Ejecutivo y Marco Decisional

En estricto cumplimiento de las directivas CX-0150 y combinando los hallazgos validados de **AG-0141** (selector de desacuerdos supervisado sobre jerarquía de modelos) y **AG-0143** (apertura virgen de confirmación y prueba de que modelos aislados sin priores históricos no superan el baseline):

1. **Construcción 100% por ID:** Los tres candidatos fueron generados mapeando individualmente cada fila por su identificador de ítem `ID` contra el ancla verificada EVT-143, asegurando cardinalidad exacta ($N = 30,000$), unicidad de claves, ausencia de valores nulos y etiquetas restringidas exclusivamente a `{"new", "used"}`.
2. **Imposibilidad Determinística para $K < 51$:** Bajo el tamaño de leaderboard público $N_{\text{pub}} = 20,998$, un candidato con $K$ modificaciones puede añadir como máximo $+K$ aciertos si el 100% de los giros es correcto y todos caen en el conjunto público:
   $$\Delta S_{\max} = \frac{K}{20,998} \implies K_{\min} = \lceil (0.95000 - 0.94761) \times 20,998 \rceil = 51$$
   Por ende, candidatos con $K \le 50$ son **matemáticamente incapaces** de alcanzar $0.95000$, independientemente de su precisión.
3. **Estructura de la Cartera de Tres Niveles:**
   - **Candidato A (Conservative Feasible, $K=75$):** Menor $K$ redondo que supera la cota matemática ($S_{\max} = 0.95118$). Alta precisión observada ($85.3\%$), pero requiere una precisión pública casi perfecta del $99.1\%$ ($52$ aciertos de los $52.5$ giros esperados en público) para llegar a $+51$.
   - **Candidato B (Medium Support, $K=160$):** Equilibrio óptimo entre soporte y precisión. Maximiza la protección a la baja (percentil 5 de ganancia es $+28.0$ aciertos netos). Score esperado de $0.94996$ (+49.3 aciertos). Requiere precisión empírica del $73.2\%$ ($82$ aciertos de $112$ en público). Alta diversificación (152 vendedores y 134 categorías).
   - **Candidato C (Broad Correction, $K=500$):** Cobertura masiva con ensemble descorrelacionado. Requiere la menor precisión empírica ($57.4\%$, $201$ aciertos de $350$ en público) para cruzar $0.95000$. Score esperado de $0.95061$ (+63.0 aciertos). Presenta la mayor probabilidad empírica conservadora ($11.29\%$), pero mayor varianza a la baja ($P_{05} = +16.0$ aciertos).
4. **Veredicto Honesto de Calibración:** Ningún candidato alcanza la meta aspiracional de $\ge 95\%$ de certeza model-based. La cota distribution-free permanece en **0.0%**. No se autoriza ninguna afirmación de certeza absoluta.

---

## 2. Auditoría Criptográfica y Estructural Completa de la Cartera

A continuación se presenta la tabla exhaustiva de auditoría por candidato, comparando hashes criptográficos, conteos de dirección, concentración y cotas determinísticas:

| Métrica / Propiedad | Candidato A: Conservative Feasible ($K=75$) | Candidato B: Medium Support ($K=160$) | Candidato C: Broad Correction ($K=500$) |
| :--- | :--- | :--- | :--- |
| **Rol Estratégico** | Menor $K$ matemáticamente viable ($S_{\max} \ge 0.95000$); máxima precisión observada | Balance óptimo; maximiza el percentil 5 de ganancia mínima; alta diversificación | Máximo soporte de corrección; menor precisión requerida; mayor probabilidad conservadora |
| **Ruta del CSV** | [`submissions/final_portfolio_ag0144_20261110/candidate_conservative_feasible_k75.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/final_portfolio_ag0144_20261110/candidate_conservative_feasible_k75.csv) | [`submissions/final_portfolio_ag0144_20261110/candidate_medium_support_k160.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/final_portfolio_ag0144_20261110/candidate_medium_support_k160.csv) | [`submissions/final_portfolio_ag0144_20261110/candidate_broad_correction_k500.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/final_portfolio_ag0144_20261110/candidate_broad_correction_k500.csv) |
| **SHA-256 del CSV** | `859e92834053ef71be7e04fb9198434d1fb5c0e0e2c3a60570e959231831f380` | `42e6c79bbc1eb9181af62ef92fd83bcf9c3667153fb749eb18268fbffb213c93` | `63a073d7b4e4ccb8d28a6995aa64b2482b805459d56e77e506cc5caf3e9150eb` |
| **Hash ID/Target** | `d65a7999692a78b3d765352a105134b9280c9c71c85954555bc59107ad86f29c` | `fb44b4b5158fba06f8ff722d590766a034396bb8b87be5e642805a6c65af6f44` | `acc515b8bcb7f7619e372e3c82d954951ac074e571a793002728b82977dbdad6` |
| **Hash SHA-256 Ancla** | `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8` | `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8` | `3a0eeba2665f63a48202c27b29ca4f1d9af88c39ecec40ed80bad4b7b08cddb8` |
| **Diff Hash vs Ancla** | `8b3cc4e3e06f11cba504f62971f0b2f6d04d04d6950ea8ba7d8212064ca20af4` | `81386d59e06beb3492ac31af163b41039e1e7e393573fb91122dd197ea06483f` | `df86519e50d0e02be4836a84675bbdd8de6cfc2c23e62afabbdff00ae481da48` |
| **Flips Exactos ($K$)** | **75** | **160** | **500** |
| **Dirección `used -> new`** | 73 (97.3%) | 154 (96.2%) | 418 (83.6%) |
| **Dirección `new -> used`** | 2 (2.7%) | 6 (3.8%) | 82 (16.4%) |
| **Vendedores Afectados** | 69 vendedores | 152 vendedores | 456 vendedores |
| **Categorías Afectadas** | 64 categorías | 134 categorías | 389 categorías |
| **Overlap EVT-143 (0.94761)**| 99.75% (29.925 / 30.000) | 99.47% (29.840 / 30.000) | 98.33% (29.500 / 30.000) |
| **Overlap EVT-135 (0.94752)**| 99.73% (29.919 / 30.000) | 99.45% (29.834 / 30.000) | 98.31% (29.494 / 30.000) |
| **Overlap EVT-132 (0.94752)**| 99.74% (29.921 / 30.000) | 99.45% (29.836 / 30.000) | 98.32% (29.496 / 30.000) |
| **Overlap EVT-130 (0.94738)**| 99.64% (29.891 / 30.000) | 99.35% (29.806 / 30.000) | 98.26% (29.478 / 30.000) |
| **Score Matemático Máximo ($S_{\max}$)** | **0.95118** (Factible) | **0.95523** (Factible) | **0.97142** (Factible) |
| **$c_{\min}$ Requerido (Todo Público)** | 63 fixes / 75 (84.0%) | 106 fixes / 160 (66.2%) | 276 fixes / 500 (55.2%) |
| **$\mathbb{E}[K_{\text{pub}}]$ Esperado en Público** | 52.5 flips | 112.0 flips | 350.0 flips |
| **$c_{\min}$ Requerido ($\mathbb{E}[K_{\text{pub}}]$)** | 52 fixes / 52.5 (**99.1%**) | 82 fixes / 112.0 (**73.2%**) | 201 fixes / 350.0 (**57.4%**) |
| **Precisión Confirm Observada** | **85.3%** ($64$ fixes, $11$ breaks) | **72.0%** ($115$ fixes, $45$ breaks) | **59.0%** ($295$ fixes, $205$ breaks) |
| **IC 95% Seller Bootstrap** | `[77.0%, 91.9%]` | `[65.0%, 79.0%]` | `[53.0%, 65.0%]` |
| **IC 95% Category Bootstrap** | `[77.0%, 92.9%]` | `[65.5%, 79.5%]` | `[53.5%, 65.5%]` |
| **Aciertos Netos Públicos Esperados** | **+37.1 aciertos** | **+49.3 aciertos** | **+63.0 aciertos** |
| **Score Público Esperado** | **0.94938** | **0.94996** | **0.95061** |

---

## 3. Simulación Conjunta Finita de Monte Carlo (100.000 Réplicas)

Para modelar con máxima fidelidad la distribución del score público real, se diseñó una simulación conjunta de **100.000 réplicas apareadas** que evalúa simultáneamente a los tres candidatos sobre exactamente las mismas condiciones de partición y de dificultad macro:

### Metodología de la Simulación:
1. **Muestreo Hipergeométrico sin Reemplazo:** En cada réplica $m \in \{1, \dots, 100.000\}$, la cantidad de flips que caen en el conjunto público ($N_{\text{pub}} = 20,998$ de $N_{\text{total}} = 30,000$) se muestrea mediante la distribución hipergeométrica exacta:
   $$K_{\text{pub}}^{(m)} \sim \text{Hypergeometric}(N_{\text{pub}} = 20,998, N_{\text{priv}} = 9,002, K)$$
2. **Dependencia Clustered Correlacionada (Sellers y Categorías):** Para capturar el riesgo de concentración de vendedores y categorías comunes, se genera un shock latente macro estandarizado $Z^{(m)} \sim \mathcal{N}(0, 1)$ común a la réplica, que induce una precisión efectiva correlacionada:
   $$\theta_k^{(m)} = \text{clip}\left(\mu_k + \sigma_k \cdot Z^{(m)}, 0, 1\right)$$
   donde $\sigma_k$ se calibra para que el intervalo $1.96 \cdot \sigma_k$ coincida con el semi-ancho del intervalo bootstrap por vendedor.
3. **Ganancia Neta en Leaderboard:** 
   $$F_k^{(m)} \sim \text{Binomial}\left(K_{\text{pub}, k}^{(m)}, \theta_k^{(m)}\right), \quad G_k^{(m)} = 2 F_k^{(m)} - K_{\text{pub}, k}^{(m)}$$
   Condición de éxito: $G_k^{(m)} \ge +51 \iff S_k^{(m)} \ge 0.95000$.

### Resultados de la Simulación Conjunta (100.000 Réplicas):

| Métrica de Desempeño | Candidato A ($K=75$) | Candidato B ($K=160$) | Candidato C ($K=500$) |
| :--- | :---: | :---: | :---: |
| **Ganancia Neta Media ($\mathbb{E}[G]$)** | $+37.1$ aciertos | $+49.3$ aciertos | **$+63.1$ aciertos** |
| **Score Público Esperado ($\mathbb{E}[S]$)** | $0.94938$ | $0.94996$ | **$0.95061$** |
| **Percentil 5 de Ganancia ($P_{05}$)** | $+25.0$ aciertos | **$+28.0$ aciertos** | $+16.0$ aciertos |
| **Mediana de Ganancia ($P_{50}$)** | $+37.0$ aciertos | $+50.0$ aciertos | **$+63.0$ aciertos** |
| **Percentil 95 de Ganancia ($P_{95}$)** | $+48.0$ aciertos | $+70.0$ aciertos | **$+110.0$ aciertos** |
| **$P(S \ge 0.95000)$ Media Central** | $2.28\%$ | $46.94\%$ | **$67.20\%$** |
| **$P(S \ge 0.95000)$ Conservadora ($Z = -1.645$)** | $0.04\%$ | $7.78\%$ | **$11.29\%$** |

### Comparaciones Pareadas Directas (Exactamente las Mismas Réplicas):
- **Candidato B vs Candidato A:**
  $$P(\text{Ganancia}_B > \text{Ganancia}_A) = \mathbf{83.31\%} \quad (\Delta \text{ media} = +12.2 \text{ aciertos públicos})$$
- **Candidato C vs Candidato B:**
  $$P(\text{Ganancia}_C > \text{Ganancia}_B) = \mathbf{70.21\%} \quad (\Delta \text{ media} = +13.7 \text{ aciertos públicos})$$
- **Candidato C vs Candidato A:**
  $$P(\text{Ganancia}_C > \text{Ganancia}_A) = \mathbf{83.59\%} \quad (\Delta \text{ media} = +26.0 \text{ aciertos públicos})$$

---

## 4. Supuestos Críticos para el Salto Confirm $\to$ Test Público

Para que las estimaciones probabilísticas de la simulación sean válidas en el test público real, se requieren formalmente los siguientes supuestos:

1. **Supuesto de Muestreo de la Partición Pública:** El leaderboard público de $20,998$ ítems se asume como una muestra aleatoria simple sin reemplazo de los $30,000$ ítems de test (hipergeométrico $\pi \approx 0.69993$). Si la partición pública estuviera sesgada por fecha, vendedor o categoría, los aciertos esperados se desviarían proporcionalmente.
2. **Supuesto de Intercambiabilidad Clustered:** La tasa de aciertos observada en los pliegues de validación de vendedores agrupados es intercambiable con la tasa de aciertos en los vendedores de test. Las variaciones por vendedor fueron acotadas mediante el remuestreo por bloques de vendedores.
3. **Invarianza de Causalidad del Ancla EVT-143:** Los errores corregidos en el ancla son verdaderas discrepancias con la etiqueta real $y_{\text{true}}$ y no artefactos causados por correlaciones espurias de palabras frecuentes en títulos.
4. **Estado Conformal Distribution-Free:** Dado que en la auditoría de procedencia prequential retrospectiva (AG-0134 / AG-0137) se determinó que no existen eventos prequentiales históricos con predicciones inmutables registradas antes del score con cobertura conformal calibrada, **la garantía matemática libre de distribución permanece estrictamente en $0.0\%$**.

---

## 5. Selección de Candidato Primario y Recomendación de Fallback Seguro

### Selección del Candidato Primario:
Bajo la regla de decisión obligatoria de CX-0150 (*"Elegí como candidato primario el de mayor probabilidad empírica conservadora de superar 0.95000, no el de mayor score medio"*):
- **Candidato Primario por Probabilidad Conservadora:** **`candidate_broad_correction_k500`** ($P_{\text{cons}} = 11.29\%$). Requiere únicamente una tasa de aciertos del $57.4\%$ en público para alcanzar $+51$ aciertos netos, ofreciendo un score esperado de **$0.95061$**.
- **Candidato Alternativo por Máxima Seguridad de Piso:** Si el objetivo fuera minimizar el riesgo de degradación a la baja, **`candidate_medium_support_k160`** ofrece el mayor percentil 5 de ganancia ($P_{05} = +28.0$ aciertos frente a $+16.0$ de C), con un score esperado de $0.94996$ y una probabilidad conservadora del $7.78\%$.
- **Candidato A ($K=75$) queda desestimado:** Aunque su precisión confirm es alta ($85.3\%$), su volumen de giros es insuficiente; requiere que $52$ de los $52.5$ giros en público sean aciertos ($99.1\%$), lo que deja su probabilidad de éxito en apenas $0.04\%$.

### Declaración Formal de Fallback Seguro:
> **DECLARACIÓN EXPLÍCITA DE INCERTIDUMBRE:**  
> Ninguno de los tres candidatos alcanza una probabilidad alta defendible ($\ge 95\%$ o $\ge 90\%$) bajo análisis de sensibilidad agrupada. Las probabilidades conservadoras se sitúan entre el $7.8\%$ y el $11.3\%$, y la cota distribution-free es $0.0\%$.  
>  
> **RECOMENDACIÓN OPERATIVA:**  
> Se recomienda **mantener como fallback seguro el ancla EVT-143 ([`submission_slot5_surgical_quad_rank6_breakthrough.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv), score `0.94761`) sin realizar envíos a Kaggle**. La cartera queda formalmente congelada en staging local para revisión y auditoría adversarial independiente por parte de Codex.

---

## 6. Verificación de Cierre y Manifiesto de Archivos Generados

- Manifiesto completo serializado: [`submissions/final_portfolio_ag0144_20261110/final_portfolio_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/final_portfolio_ag0144_20261110/final_portfolio_manifest.json) (SHA-256: `1ef3bb6832b982db57f0707d670d290921ac3abf208769566775ed7c516b56da`)
- Reporte analítico: [`results/final_portfolio_ag0144_20261110/final_portfolio_ag0144_report.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/final_portfolio_ag0144_20261110/final_portfolio_ag0144_report.json)
- Script ejecutor verificado: [`scripts/generate_final_portfolio_ag0144.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/generate_final_portfolio_ag0144.py)

**Cuota Kaggle:** Confirmamos el consumo de **0 envíos y 0 llamadas a la API de Kaggle** en toda la sesión. La tanda queda formalmente cerrada y lista para la auditoría de Codex.
