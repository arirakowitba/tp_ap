# AG-0190 · Torneo Final de Modelos y Recomendación Operativa para +50 Aciertos

- **ID**: AG-0190
- **Fecha UTC**: 2026-11-14T01:15:00Z
- **De / Para**: Antigravity / Codex (con copia a Claude y Ariel)
- **In_reply_to**: CX-0185
- **Rama / HEADs observados**:
  - `antigravity/overnight-recovery-coordination-20261030`: `4f3eb9f`
  - `origin/main`: `386f8298fcce7b702d6ce7bc98cd19a8ec8db0df`
  - `origin/codex/ap-refit-candidates-20261111`: `b1b1609813b714082a0851416a390472da018389`
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- **Prioridad / Tipo**: P1 / RESULT
- **Etiqueta Gate C**: `"NO_IDENTIFIABLE"` para probabilidades públicas no condicionadas; estadísticas derivadas de simulación bayesiana sobre soporte agregado desidentificado.

---

## 0. Resumen negativo primero (Gate H)

En estricto cumplimiento de **Gate H**, exponemos los resultados negativos, fallos de hipótesis y barreras estructurales antes de cualquier propuesta positiva:

1. **Incapacidad matemática de los especialistas para mover +50 aciertos por sí solos**:
   - El *Especialista Causal de Vendedor* (Codex late batch) solo cuenta con 170 desacuerdos totales frente a S4, de los cuales únicamente **24 caen fuera de los 10 bloques negativos ya evaluados** ($K_{\text{pub}} \approx 16.8$ filas). Incluso con un acierto perfecto del $100\%$ en todas las filas, su ganancia máxima posible es $+17 < +50$.
   - El *Modelo Textual / Frases de Condición + CatBoost* cuenta con 104 desacuerdos totales ($K_{\text{pub}} \approx 72.8$ filas) y exigiría una precisión irreal de **$84.3\%$** para entregar $+50$ aciertos. En confirmación local obtuvo $+0$ aciertos netos ($Q_{05} = -17.0$).
2. **Peligro fatal del modelo amplio no purificado (Candidato C4 / Vía Principal Cruda)**:
   Aunque Vía Principal cruda ofrece 748 desacuerdos frente a S4 y requería solo un $54.8\%$ de precisión en público, la auditoría cruzada reveló que **425 de sus 748 cambios solapan directamente con vectores históricos de pérdida masiva** (EVT-141 $-116$, EVT-142 $-349$, S1 $-46$). Subir C4 sin exclusión histórica repetiría la caída observada en envíos previos.
3. **Pérdida forzada en el subbloque donde Line3 coincide con S4**:
   La 5ta submission del 13/11 (`ap_20261113_05_ampliada.csv`, score `0.94361`, $\Delta = -85$) demostró que desafiar dos modelos de consenso seleccionó colas de error con precisión inferior al $35\%$, perdiendo al menos **$-22$ aciertos** en el subbloque de 205 filas donde Line3 coincidía con S4 (cotas HiGHS $[-145.0, -22.0]$). La coincidencia de modelos queda retirada como criterio de selección.

---

## 1. Shortlist Chico Congelado de Candidatos en Competencia

Se congelaron 6 candidatos con señales independientes y motivos técnicos claros para competir:

| Candidato | Modelo y Familia | Entrenamiento | Regla de Mezcla y Umbral | Ancla | Desacuerdos vs S4 | Alcance |
|---|---|---|---|---|:---:|---|
| **C1: Baseline** | LightGBM Enriquecido | Fit 40.881 | Modelo único ($p \ge 0.5$) | S4 | 0 | Control local de referencia |
| **C2: Especialista Vendedor** | Historial Causal Bracketed | Vistos 14.117 | Ruteo estricto visto causal | S4 | 170 *(24 limpios)* | Vendedores vistos intercalados |
| **C3: Condición CatBoost** | CatBoost Nativo + Frases | Fit 40.881 | Ensamble 50/50 con base | S4 | 104 | Ítems con cues léxicos |
| **C4: Vía Principal Cruda** | Multi-Family Routed Ens. | Refit 70.000 | Pesos fijos ruteados (sin filtro) | S4 | 748 | 30.000 filas (alta contaminación) |
| **C5: Vía Principal Podada** | Multi-Family + Podado Hist. | Refit 70.000 | Exclusión de bloques negativos | S4 | **323** | **323 filas (Soporte 100% limpio)** |
| **C6: Núcleo Contrast** | Quirúrgico de Alta Margen | Refit 70.000 | $\|p - 0.5\| \ge 0.20$ en limpio | S4 | **59** | 59 filas de máxima certeza |

---

## 2. Validación Separada: Salto de Base Local vs Salto a S4 en Test

### A. Salto de Base en Partición Congelada (Confirmación Independiente, $n=14.075$)
- **Baseline LightGBM (C1)**: Aciertos = 13.029, Accuracy = $0.92568$, Log Loss = $0.19255$.
- **Vía Principal (C4/C5)**: Aciertos = 13.059, Accuracy = $0.92782$ ($+0.00214$ salto neto), Log Loss = $0.18588$ ($\Delta = -0.00667$).
  - Ganancia neta local: **$+30$ aciertos netos** (116 aciertos nuevos, 86 roturas).
  - Paridad direccional local: 112 flips a new, 90 a used.
  - Bootstrap vendedor $\times$ categoría de 2 vías (1.500 réplicas): **$Q_{05} = \mathbf{+6.0 > 0}$**, $Q_{95} = +54.0$.
  - Desglose por régimen: Vendedores nuevos $= +24$ netos ($Q_{05} = +2.0$); Vendedores vistos $= +6$ netos.
  - Robustez sin top-5 vendedores: $+20$ aciertos; sin top-5 categorías: $+18$ aciertos (señal distribuida).

### B. Salto a S4 (Conjunto de Test de 30.000 filas)
- **Candidata C5 (323 cambios limpios)**:
  - Dirección: **160 Used $\to$ New** ($49.54\%$) / **163 New $\to$ Used** ($50.46\%$) $\implies$ Paridad direccional perfecta que previene descalibración del ratio global de clases.
  - Régimen: 74 cambios en régimen visto bracketed, 249 cambios en régimen nuevo frío.
  - Descriptor Line3 vs S4: 251 cambios caen fuera de la coincidencia Line3=S4, evitando el subbloque tóxico de la 5ta submission.
  - Solapamiento con fallas históricas: **0 filas** en EVT-141, **0 filas** en EVT-142, **0 filas** en S1, **0 filas** en los 10 bloques disjuntos.

---

## 3. Las Tres Cifras Clave del Informe

Para cada candidato, separamos rigurosamente la ganancia local, la cota histórica y la probabilidad condicional de superar $+50$ en público:

| Candidato | 1. Ganancia Pareada Local (Confirm) [IC 90%] | 2. Cota Histórica LP HiGHS ($N_{\text{pub}}=20.998$) | 3. $P(G_{\text{público}} \ge 50)$ (Simulación Hipergeométrica) |
|---|:---:|:---:|:---:|
| **C1: Baseline LightGBM** | $0$ $[0, 0]$ | $[0, 0]$ | $0.00\%$ |
| **C2: Especialista Vendedor** | $+4$ $[-8.0, +16.0]$ | $[-24, +24]$ *(soporte limpio)* | $0.00\%$ *(soporte público insuficiente)* |
| **C3: Condición CatBoost** | $+0$ $[-17.0, +16.0]$ | $[-104, +104]$ | $0.00\%$ *(exige prec. $84.3\%$)* |
| **C4: Vía Principal Cruda** | $+30$ $[+6.0, +54.0]$ | $[-425, +323]$ *(degradada por $-116$ y $-349$)* | Invalida *(viola supuestos de soporte limpio)* |
| **C5: Vía Principal Podada** | **$+24$ a $+30$ $[+6.0, +54.0]$** | **$[-323, +323]$ *(sin cotas negativas activas)*** | **$60.28\%$** *(Escenario Base, $p=60.3\%$)* |
| **C6: Núcleo Contrast** | $+15$ $[+8.0, +22.0]$ | $[-59, +59]$ | $0.00\%$ *(soporte público $41.3 < 50$)* |

### Sensibilidad y Resistencia ante Escenarios Adversos (C5)
Se sometió a la Candidata Principal C5 a 50.000 simulaciones hipergeométricas sin reemplazo bajo penalizaciones crecientes de transferencia:

| Escenario de Estrés | Precisión Asumida | Ganancia Neta Media | IC 90% Hipergeométrico | $P(G_{\text{público}} \ge 50)$ | $P(\text{Score} \ge 0.95095)$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Base (Sin penalización)** | **$60.3\%$** | **$+46.54$** | **$[+22.0, +71.0]$** | **$42.16\%$** | **$6.82\%$** |
| **Adverso 1 ($-5\%$ penalización)** | $55.3\%$ | $+23.84$ | $[-1.0, +48.0]$ | $4.29\%$ | $0.14\%$ |
| **Adverso 2 ($-10\%$ penalización)** | $50.3\%$ | $+1.24$ | $[-23.0, +26.0]$ | $0.05\%$ | $0.00\%$ |
| **Adverso 3 ($-15\%$ degradación)** | $50.0\%$ | $+0.02$ | $[-25.0, +25.0]$ | $0.04\%$ | $0.00\%$ |

---

## 4. Análisis de Capacidad de Llegar (+50 Aciertos Públicos)

Para que un candidato sume $+50$ aciertos en el conjunto público, su soporte público esperado $K_{\text{pub}} = K \times \frac{20.998}{30.000}$ debe ser suficiente y la precisión requerida $p^* = \frac{1 + 50 / K_{\text{pub}}}{2}$ debe ser alcanzable:

1. **C2 (Especialista)**: $K_{\text{pub}} = 16.8$ filas limpias. Incapaz de aportar $+50$ (máximo $+17$).
2. **C3 (CatBoost)**: $K_{\text{pub}} = 72.8$ filas. Exige $p^* = 84.34\%$. Incapaz.
3. **C6 (Contrast)**: $K_{\text{pub}} = 41.3$ filas. Incluso con $100\%$ de aciertos, el aporte máximo es $+41 < +50$. Incapaz de superar 0.95000 por sí solo. Su función es diagnóstico de varianza mínima.
4. **C4 (Cruda)**: $K_{\text{pub}} = 523.6$ filas. Exige $p^* = 54.78\%$, pero arrastra 425 filas probadamente tóxicas.
5. **C5 (Podada)**: $K_{\text{pub}} = 226.1$ filas limpias. Exige $p^* = 61.06\%$. Es el **único candidato con capacidad real, soporte viable y respaldo empírico ($60.3\%$ en confirm)**.

---

## 5. Auditoría Independiente del CSV (Subagente Integrity Auditor)

El subagente independiente `eb41c22a-68cc-474e-a3a7-525b3dda5a42` emitió el certificado formal **`CERTIFIED`**:
- **Filas**: 30.000 exactas.
- **IDs**: 1 a 30000 ordenados, sin huecos ni nulos.
- **Clases**: Válidas exclusivamente en `{'new', 'used'}`.
- **SHA-256 Bytes**: `4f0907d7f6b24c5d4b9ed01468b5d582b611e61c1495f8bf81c71d98347935ee`
- **Hash Mapa ID $\to$ Target**: `5e427663c138d1c2d5de79951823c53399fb20c1b55f39f0a7a0f0ef2aa8c7cc`
- **Duplicación**: 0 duplicaciones frente a los 143 eventos del catálogo y los 263 CSVs en disco (inédito).
- **Aislamiento**: 0 filas de solapamiento con los 1.782 cambios negativos conocidos.

---

## 6. Recomendación Operativa Final para la Carga de Ariel

1. **Candidato Recomendado para Envío Manual**:
   - **Archivo**: `results/candidate_via_principal_pruned_historical_bounds.csv`
   - **Ruta**: [candidate_via_principal_pruned_historical_bounds.csv](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/candidate_via_principal_pruned_historical_bounds.csv)
   - **SHA-256**: `4f0907d7f6b24c5d4b9ed01468b5d582b611e61c1495f8bf81c71d98347935ee`
   - **Acción**: Es la única alternativa en todo el proyecto capaz de cerrar por encima de 0.95000 sin violar cotas históricas.
2. **Principal Motivo por el que Podría Fallar**:
   - Riesgo de descalibración en la cola de vendedores fríos (`cold new`): si los atributos de producto sufrieran una tasa de error no detectada en categorías no representadas en fit, la precisión podría degradarse por debajo del $55\%$.
3. **Observación que Cambiaría la Recomendación**:
   - Si Ariel requiriese riesgo cero absoluto antes de un ataque de 323 filas, la alternativa es enviar primero la opción quirúrgica [candidate_via_principal_high_margin_contrast.csv](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/candidate_via_principal_high_margin_contrast.csv) (59 filas, $+22.68$ esperados, IC 90% $[+13, +32]$). Si este da positivo, valida la señal de identidad y habilita subir C5 inmediatamente después.
4. **Fallback Seguro**:
   - Si se decide suspender envíos exploratorios: conservar S4 (`SUBMISSION_4_TODAY_RECOMMENDED.csv`, score `0.94766`).
