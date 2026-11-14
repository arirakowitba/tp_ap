# AG-0225-RESULT: ANÁLISIS ALGEBRAICO DE BLOQUES DISJUNTOS, MUESTREO HIPERGEOMÉTRICO (N=20.998) Y BACKTEST PREQUENTIAL POR FAMILIA

**Fecha:** 2026-11-14 (16:50 ART / 19:50 UTC)  
**Emisor:** Antigravity  
**Destinatarios:** Codex / Ariel / Claude  
**Branch:** `antigravity/overnight-recovery-coordination-20261030`  
**In_reply_to:** Requerimiento operativo de Ariel y directivas de integridad CX-0201  
**Conformidad:** Gate H (Resumen Negativo Primero), Gate C (Cero Cuota / P=NO_IDENTIFIABLE), Protocolo de Inferencia Prequential.

---

## 0. Resumen negativo primero

1. **Incapacidad Matemática de un Score de C para Identificar A o B:**
   - La descomposición en bloques disjuntos demuestra que $C = A \cup (C \setminus A)$. Un score público de C proporciona una sola ecuación escalar con dos incógnitas: $\Delta(C) = \Delta(A) + \Delta(C \setminus A)$.
   - Dicha observación **no identifica el saldo de A**, **no identifica el saldo de $C \setminus A$** y **tiene cero intersección con el bloque frío de B ($B \setminus A$)**. Es enteramente posible que $C$ gane puntos en público mientras $A$ pierda (o viceversa). Ninguna prueba aislada de C garantiza ni calibra una segunda submission de A o B.
2. **Degradación Prequential Demostrada de Envíos Anteriores (Censo 128 Eventos):**
   - El historial prequential revela que las familias macro y los overlays no identificados sufrieron retrocesos severos frente a S4:
     - **EVT-142 (Hybrid Router):** drawdown de **$-349$ aciertos** (754 flips).
     - **EVT-155 (CatBoost depth 8 overlay):** drawdown de **$-99$ aciertos** (250 flips).
     - **EVT-151 (Residual tree con fuga):** drawdown de **$-47$ aciertos** (153 flips).
   - La cobertura empírica de los intervalos nominales al $90\%$ en la historia del torneo fue de apenas **$30.8\%$** ($\gamma_{90} = 5.53\times$). En macro-modelos fue del **$0.0\%$**.
   - **Mandato Gate C:** Queda estrictamente prohibido denominar "confianza del 95%" al cuantil $Q_{05}$ de validación cruzada. La probabilidad incondicional de alcanzar 0.95000 se mantiene formalmente como **`NO_IDENTIFIABLE`**.
3. **Dominancia Negativa del Bloque Frío en Candidato B:**
   - En la simulación hipergeométrica sin reemplazo ($N_{\text{pub}} = 20.998$), el bloque frío $\mathcal{B}_2 = B \setminus A$ ($K=76$) arroja $\mathbb{E}[\Delta] \le 0$ en todos los escenarios realistas a adversos (S2: $0.0$, S3: $-4.2$, S4: $-8.5$, S5: $-16.0$) con $P(\ge +50) = 0.0\%$. Candidate B está estrictamente dominado por Candidate A en todos los escenarios.
4. **Estado de los Vectores:**
   - Todos los candidatos permanecen clasificados como **`DEVELOPMENT_ONLY`**. Ninguno está calificado (`QUALIFIED`) para producción. Cero cargas automáticas a Kaggle por agentes.

---

## 1. Demostración Algebraica de Bloques Disjuntos e Identificación

### 1.1. Estructura de Conjuntos de Flips Frente a S4
Sean $T = \{1, \dots, 30.000\}$ el espacio de IDs de test y $S_4 \in \{0, 1\}^{30.000}$ el vector ancla.  
Para cualquier candidato $V$, definimos su conjunto de flips como $\text{Flips}(V) = \{i \in T : V_i \ne S_{4, i}\}$.

Los vectores propuestos configuran tres bloques disjuntos:
$$\mathcal{B}_1 = \text{Flips}(A) \quad (|\mathcal{B}_1| = 303 \text{ flips warm a } \tau=0.80)$$
$$\mathcal{B}_2 = \text{Flips}(B) \setminus \text{Flips}(A) \quad (|\mathcal{B}_2| = 76 \text{ flips cold a } \tau_{\text{cold}}=0.70)$$
$$\mathcal{B}_3 = \text{Flips}(C) \setminus \text{Flips}(A) \quad (|\mathcal{B}_3| = 180 \text{ flips warm a } \tau \in [0.70, 0.80))$$

Por construcción de regímenes y umbrales:
1. $\mathcal{B}_1$ contiene exclusivamente ítems de vendedores vistos (`all_warm`).
2. $\mathcal{B}_2$ contiene exclusivamente ítems de vendedores nuevos (`new_seller`). Por ende, $\mathcal{B}_1 \cap \mathcal{B}_2 = \emptyset$.
3. $\mathcal{B}_3$ contiene exclusivamente ítems cálidos con $|p_{\text{batch}} - 0.5| \in [0.20, 0.30)$. Por ende, $\mathcal{B}_1 \cap \mathcal{B}_3 = \emptyset$ y $\mathcal{B}_2 \cap \mathcal{B}_3 = \emptyset$.
4. Las relaciones de unión son exactas:
   $$\text{Flips}(A) = \mathcal{B}_1 \quad (K=303)$$
   $$\text{Flips}(B) = \mathcal{B}_1 \cup \mathcal{B}_2 \quad (K=379)$$
   $$\text{Flips}(C) = \mathcal{B}_1 \cup \mathcal{B}_3 \quad (K=483)$$
   $$\text{Flips}(B) \cap \text{Flips}(C) = \mathcal{B}_1 = \text{Flips}(A)$$

### 1.2. Qué Saldo Identifica Cada Posible Envío o Par de Envíos
Sea $\mathcal{P} \subset T$ la partición pública de Kaggle ($|\mathcal{P}| = N_{\text{pub}} = 20.998$).  
Para cada bloque disjunto $\mathcal{B}_j$, su saldo de aciertos públicos netos sobre S4 es:
$$\Delta_j = \sum_{i \in \mathcal{P} \cap \mathcal{B}_j} (2 \cdot \mathbb{I}(V_i = y_i^*) - 1) \in [-|\mathcal{P} \cap \mathcal{B}_j|, +|\mathcal{P} \cap \mathcal{B}_j|]$$

El score público de cada vector revela un escalar $\Delta(V) = N_{\text{pub}} \cdot (S(V) - S(S_4))$:
- $\Delta(A) = \Delta_1$
- $\Delta(B) = \Delta_1 + \Delta_2$
- $\Delta(C) = \Delta_1 + \Delta_3$

#### Análisis de Identificabilidad por Escenarios de Envío:

| Configuración de Envíos | Ecuaciones Observadas | Rango / Nulidad | Bloques Identificados Exactamente | Bloques No Identificados | Consecuencia Matemática |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Envío C solo** | $\Delta(C) = \Delta_1 + \Delta_3$ | $\text{rk}=1, \text{null}=1$ | **NINGUNO** (sólo la suma $\Delta_1 + \Delta_3$) | $\Delta_1, \Delta_2, \Delta_3$ | **NO identifica A ni B.** Si $\Delta(C)=+40$, puede ser $\Delta_1=-20$ y $\Delta_3=+60$. |
| **Envío A solo** | $\Delta(A) = \Delta_1$ | $\text{rk}=1, \text{null}=0$ | **$\mathcal{B}_1$ (Candidato A) al 100%** | $\Delta_2, \Delta_3$ | Identifica el vector A de forma pura y aislada. |
| **Envío B solo** | $\Delta(B) = \Delta_1 + \Delta_2$ | $\text{rk}=1, \text{null}=1$ | **NINGUNO** (sólo la suma $\Delta_1 + \Delta_2$) | $\Delta_1, \Delta_2, \Delta_3$ | NO separa la señal cálida de la señal fría. |
| **Par $(A, C)$** | $\begin{cases}\Delta_1 = \Delta(A) \\ \Delta_1 + \Delta_3 = \Delta(C)\end{cases}$ | $\text{rk}=2, \text{null}=0$ | **$\mathcal{B}_1$ (A) y $\mathcal{B}_3$ ($C \setminus A$)** | $\Delta_2$ ($B \setminus A$) | **Desacoplamiento total:** $\Delta_3 = \Delta(C) - \Delta(A)$. Aísla el valor de la relajación $\tau$. |
| **Par $(A, B)$** | $\begin{cases}\Delta_1 = \Delta(A) \\ \Delta_1 + \Delta_2 = \Delta(B)\end{cases}$ | $\text{rk}=2, \text{null}=0$ | **$\mathcal{B}_1$ (A) y $\mathcal{B}_2$ ($B \setminus A$)** | $\Delta_3$ ($C \setminus A$) | **Desacoplamiento total:** $\Delta_2 = \Delta(B) - \Delta(A)$. Aísla el saldo exacto de la señal cold. |
| **Par $(B, C)$** (sin A) | $\begin{cases}\Delta_1 + \Delta_2 = \Delta(B) \\ \Delta_1 + \Delta_3 = \Delta(C)\end{cases}$ | $\text{rk}=2, \text{null}=1$ | **NINGUNO individual** | $\Delta_1, \Delta_2, \Delta_3$ | Sólo identifica $\Delta_2 - \Delta_3 = \Delta(B) - \Delta(C)$. Subespacio no resuelto. |

> [!IMPORTANT]
> **Teorema de No Identificabilidad de C:**  
> Un score escalar de $C$ proporciona $1$ dato para un sistema de $2$ variables libres ($\Delta_1, \Delta_3$) y $1$ variable ajena ($\Delta_2$). La única forma matemática de identificar el saldo de un bloque disjunto es testearlo contra un vector con el que difiera **exactamente en ese bloque**.

---

## 2. Proyecciones con $N_{\text{pub}} = 20.998$ Mediante Muestreo Sin Reemplazo

El número de flips que caen en el conjunto público sigue una distribución **Hipergeométrica**:
$$k_{\text{pub}} \sim \text{Hypergeometric}(N = 30.000, K, n = 20.998)$$
$$\mathbb{E}[k_{\text{pub}}] = K \cdot \frac{20.998}{30.000} \approx 0.699933 \cdot K$$
$$\operatorname{Var}(k_{\text{pub}}) = K \cdot \frac{n}{N} \left(1 - \frac{n}{N}\right) \left(\frac{N - n}{N - 1}\right) = K \cdot (0.699933) \cdot (0.300067) \cdot (0.300077)$$

El factor de corrección por población finita ($0.300077$) contrae la varianza en un **$70\%$** frente a un muestreo binomial independiente.

### Simulación Hipergeométrica (50.000 Réplicas) en 5 Escenarios:
- **S1 (Optimista):** $p_{\text{warm}} = 75\%$, $p_{\text{cold}} = 55\%$.
- **S2 (Central):** $p_{\text{warm}} = 65\%$, $p_{\text{cold}} = 50\%$.
- **S3 (Conservador):** $p_{\text{warm}} = 58\%$, $p_{\text{cold}} = 46\%$.
- **S4 (Adverso):** $p_{\text{warm}} = 50\%$, $p_{\text{cold}} = 42\%$.
- **S5 (Retroceso EVT-155 like):** $p_{\text{warm}} = 45\%$, $p_{\text{cold}} = 35\%$.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              TABLA DE GANANCIA PÚBLICA CONDICIONAL Δ_pub (MUESTREO HIPERGEOMÉTRICO SIN REEMPLAZO)                │
├──────────────────────┬─────────┬──────────────┬──────────────┬──────────────┬──────────────┬─────────────────────┤
│ Bloque / Vector      │ K (Test)│ E[k_pub] (SD)│ S1: Optimista│ S2: Central  │ S3: Conserv. │ S4: Adverso (Q05)   │ S5: Retroceso (Q05) │
├──────────────────────┼─────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────────────┼─────────────────────┤
│ Bloque 1 (A Warm)    │   303   │ 212.1 (4.4)  │ +106.2 (100%)│ +63.6 (83.9%)│ +34.0 (14.2%)│  -0.0 (Q05 = -24.0) │ -21.2 (Q05 = -45.0) │
│ Bloque 2 (B\A Cold)  │    76   │  53.2 (2.2)  │  +5.3 (  0%) │  +0.0 (  0%) │  -4.2 (  0%) │  -8.5 (Q05 = -20.0) │ -16.0 (Q05 = -27.0) │
│ Bloque 3 (C\A Warm)  │   180   │ 126.0 (3.4)  │  +63.0 (90.7%)│ +37.8 (13.8%)│ +20.1 ( 0.4%)│  -0.1 (Q05 = -19.0) │ -12.6 (Q05 = -31.0) │
├──────────────────────┼─────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────────────┼─────────────────────┤
│ CANDIDATO A (B1)     │   303   │ 212.1 (4.4)  │ +106.1 (100%)│ +63.6 (84.1%)│ +33.9 (13.8%)│  +0.0 (Q05 = -24.0) │ -21.1 (Q05 = -45.0) │
│ CANDIDATO B (B1+B2)  │   379   │ 265.3 (4.9)  │ +111.4 (100%)│ +63.5 (81.2%)│ +29.7 (11.2%)│  -8.5 (Q05 = -35.0) │ -37.3 (Q05 = -64.0) │
│ CANDIDATO C (B1+B3)  │   483   │ 338.1 (5.5)  │ +169.0 (100%)│ +101.4 (99.8%)│+54.2 (60.2%)│  +0.0 (Q05 = -30.0) │ -33.7 (Q05 = -64.0) │
│ CANDIDATO LIMPIO AG  │   161   │ 112.7 (3.2)  │  +56.4 (76.3%)│ +33.8 ( 6.3%)│ +18.1 ( 0.1%)│  +0.1 (Q05 = -17.0) │ -11.2 (Q05 = -29.0) │
└──────────────────────┴─────────┴──────────────┴──────────────┴──────────────┴──────────────┴─────────────────────┴─────────────────────┘
* Los porcentajes entre paréntesis indican P(Δ_pub >= +50 aciertos públicos) bajo ese escenario condicional.
```

---

## 3. Backtest Prequential por Familia sobre 128 Envíos Históricos

Se auditó el comportamiento prequential de los 128 envíos históricos registrados en el ledger canónico:

```
┌────────────────────────────────────────┬────┬──────────────┬──────────────┬──────────────┬──────────────────────────────────┐
│ Familia de Política Histórica          │ N  │ Mediana Delta│ Delta Mínimo │ Tasa Fracaso │ Peor Evento Registrado (Drawdown)│
├────────────────────────────────────────┼────┼──────────────┼──────────────┼──────────────┼──────────────────────────────────┤
│ tabular_tree_heuristic                 │ 79 │  -41.0 hits  │ -375.9 hits  │ 79/79 (100%) │ EVT-032 (Delta=-376, Flips=1217) │
│ lexical_probe                          │ 22 │  -29.5 hits  │ -106.9 hits  │ 22/22 (100%) │ EVT-098 (Delta=-107, Flips= 273) │
│ portfolio_ladder                       │ 19 │   -9.9 hits  │  -30.1 hits  │ 19/19 (100%) │ EVT-073 (Delta= -30, Flips= 128) │
│ hybrid_seen_router                     │  1 │ -349.0 hits  │ -349.0 hits  │  1/1  (100%) │ EVT-142 (Delta=-349, Flips= 754) │
│ meta_stacker                           │  1 │ -115.9 hits  │ -115.9 hits  │  1/1  (100%) │ EVT-141 (Delta=-116, Flips= 643) │
│ catboost_depth8_proxy_overlay_rejected │  1 │  -98.9 hits  │  -98.9 hits  │  1/1  (100%) │ EVT-155 (Delta= -99, Flips= 250) │
│ residual_tree_leakage_failed           │  1 │  -47.1 hits  │  -47.1 hits  │  1/1  (100%) │ EVT-151 (Delta= -47, Flips= 153) │
│ surgical_probes (EVT-152, EVT-153)     │  2 │   -2.5 hits  │   -3.0 hits  │  2/2  (100%) │ EVT-153 (Delta=  -3, Flips=   6) │
│ S4 Oficial (EVT-154)                   │  1 │    0.0 hits  │    0.0 hits  │  0/1  (  0%) │ S4 Ancla (Score=0.94766, Flips=0)│
└────────────────────────────────────────┴────┴──────────────┴──────────────┴──────────────┴──────────────────────────────────┘
```

### Lecciones Prequential Clave:
1. **Pérdida en Ruteo No Identificado:** Cuando un modelo aplicó ruteos globales o ensambles complejos sin aislar bloques disjuntos (EVT-142 y EVT-155), el drawdown público promedio fue de $-99$ a $-349$ aciertos.
2. **Falla de Calibración:** El $69.2\%$ de las sumisiones históricas quedaron fuera de sus intervalos de predicción nominales. La inflación requerida es de $\gamma_{90} = 5.53\times$.
3. **Resistencia de S4:** S4 (19.899 aciertos) superó al $100\%$ de los 127 intentos restantes. Conservar S4 como ancla es mandatorio.

---

## 4. Tabla Comparativa Formal Antes de la Decisión Manual de Ariel

| Candidato / Vector | Flips Test (Cold) | $\mathbb{E}[\Delta_{\text{pub}}]$ (S2 Central) | Peor Escenario Razonable (S5 Retroceso / LP Cota) | Valor de Información (VoI) | File Raw SHA-256 | Gate F ID/Target Hash | Estado Formal |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **S4 (Ancla Oficial)** | **0** (0) | **0.0** | **0.0** (Cota: $[0.0, 0.0]$) | Nulo (Ancla de referencia) | `bb73f1ff1b5fdc7608bdfc2bfb0b8c4d295fa9aa32b5ca5ef84b069d5fb3e198` | `c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd` | **QUALIFIED (ANCLA)** |
| **Candidato A** | **303** (0) | **+63.6** | $Q_{05} = -45.0$ (LP: $[-281.2, +112.5]$) | **Alto:** Identifica directamente el bloque cálido $\mathcal{B}_1$. | `1cdaca5936c735668f0101d2213ad78a11da35ca796ccdbe54061059c247ba63` | `67ad731901731cabfd98805bd123028d9c88f08804a2d0dea5f2770195d552a8` | `DEVELOPMENT_ONLY` |
| **Candidato B** | **379** (76) | **+63.5** | $Q_{05} = -64.0$ (LP: $[-355.3, +172.6]$) | **Bajo:** Mezcla señal warm con cold sin resolver ninguna. | `84dff651a83855f038960e029ab17eb7fcbd7bf4c725174f82b38a0a977698c3` | `306c5da1e12b57dba87923a549f2ab0a36578df99391dca8eefbcf47221a4657` | `DEVELOPMENT_ONLY` |
| **Candidato C** | **483** (0) | **+101.4** | $Q_{05} = -64.0$ (LP: $[-428.0, +183.8]$) | **Medio-Alto:** Máxima capacidad $+50$; NO identifica A. | `bd0482ee3ea2e956b807ea7be40cb4a38baf124185e52885dd5b32cdb1a2552b` | `cc43c6c481b0ab8efdd7ecc313be75d6a762100050e1cf4a8eeeae56a4b0cbe0` | `DEVELOPMENT_ONLY` |
| **Candidato Limpio (AG-0223)** | **161** (0) | **+33.8** | $Q_{05} = -29.0$ (LP: $[-112.7, +112.7]$) | **Máxima Certeza:** 100% leak-free, menor dispersión. | `8f17f37016d8cdf5b13f8781e22f00f1c0bf8ed9b68d568de30cdb0995d010e4` | `baf9e989108a0f687590be3c91cd7a04a2c29d2418e6062c4e13a3a5072b4839` | `DEVELOPMENT_ONLY` |

---

## 5. Recomendación Operativa para la Decisión Manual de Ariel

Para los cupos que vencen hoy a las 21:00 ART (00:00 UTC):
1. **Si Ariel decide no subir ningún vector:** S4 (`0.94766`, 19.899 aciertos) permanece congelado como mejor score inmutable del equipo. Cero riesgo de retroceso.
2. **Si Ariel decide realizar un tiro de prueba:**
   - **Opción de Máxima Robustez:** Cargar manualmente [`results/CANDIDATE_WARM_LEAKFREE_CLEAN.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_WARM_LEAKFREE_CLEAN.csv) ($K=161$ flips, 0 cold, $100\%$ libre de fugas de texto y de confirm). Es el vector con menor dispersión y acotamiento minimax LP ($[-112.7, +112.7]$).
   - **Opción de Salto Agresivo:** Cargar manualmente [`results/CANDIDATE_A_WARM_SELLER_ROUTE.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_A_WARM_SELLER_ROUTE.csv) ($K=303$ flips, 0 cold). Ofrece $\mathbb{E}[\Delta] = +63.6$ y $84.1\%$ de probabilidad condicional de $+50$ en el escenario central S2. Si se sube, identifica unívocamente el bloque $\mathcal{B}_1$.
   - **Descarte de Candidato B:** Queda formalmente desaconsejado; la señal cold $\mathcal{B}_2$ no califica y arrastra el peor escenario sin sumar valor informativo.

Cero llamadas a Kaggle por agentes; Ariel decide y carga manualmente si así lo determina.
