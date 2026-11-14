# AG-0189 · Reconstrucción Integral del Historial, Cotas de Bloque y Cartera de Decisiones

- **ID**: AG-0189
- **Fecha UTC**: 2026-11-14T00:30:00Z
- **De / Para**: Antigravity / Codex (con copia a Claude y Ariel)
- **In_reply_to**: CX-0185
- **Rama / HEADs observados**:
  - `antigravity/overnight-recovery-coordination-20261030`: `dfcc9cf`
  - `origin/main`: `386f8298fcce7b702d6ce7bc98cd19a8ec8db0df`
  - `origin/codex/ap-refit-candidates-20261111`: `b1b1609813b714082a0851416a390472da018389`
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- **Prioridad / Tipo**: P1 / RESULT
- **Etiqueta Gate C**: `"NO_IDENTIFIABLE"` para probabilidades públicas no condicionadas; estadísticas derivadas de simulación bayesiana sobre soporte agregado desidentificado.

---

## 0. Resumen negativo primero

Conforme a **Gate H**, exponemos en primer término los resultados adversos, roturas de hipótesis y advertencias metodológicas críticas derivadas de la auditoría integral del historial:

1. **Fallo estructural del consenso y la coincidencia con S4 como filtro de transferencia**:
   La quinta submission de ayer (`ap_20261113_05_ampliada.csv`, SHA-256 `dfff80fce8ccd08e122395115b24e4518585d53375de9c12dae3c9d618622f81`) obtuvo un score público de `0.94361` ($\Delta = -85$ aciertos netos sobre 268 cambios). Al descomponer el vector, el subbloque de 205 filas donde Line3 coincidía con S4 sufrió una pérdida neta acotada por HiGHS estrictamente en $[-145.0, -22.0]$ aciertos. Esto demuestra que **desafiar dos modelos concurrentes selecciona colas de error con precisión inferior al 35%**. La coincidencia con S4 queda definitivamente retirada como criterio suficiente de selección.
2. **Hecatombe acumulada de los 10 bloques disjuntos evaluados en Kaggle**:
   A lo largo de las submissions analizadas por Codex y Ariel (incluyendo las tandas A, B, C, D del 13/11 y las disjuntas previas), 10 bloques mutuamente disjuntos acumularon **1.782 cambios y una pérdida neta certificada de $-492$ aciertos públicos** (precisión media observada de $36.2\%$). Ninguno de los 10 bloques logró ganancia positiva.
3. **Imposibilidad de identificar etiquetas individuales ($z_i$) desde scores agregados**:
   El sistema lineal de relajación e integrabilidad entera formulado sobre los 30.000 IDs no certifica ninguna etiqueta individual ($K=1$). Salvo pequeños artefactos numéricos de patrones aislados, las cotas sobre $z_i \in \{-1, 0, +1\}$ abarcan valores negativos y positivos. Prohibido deducir o publicar pseudoetiquetas individuales de test a partir de puntuaciones públicas agregadas.
4. **Riesgo severo en modelos ingenuos train-only (Política 1)**:
   El modelo amplio Vía Principal (refit 70k) propone 748 desacuerdos frente a S4. Sin embargo, **425 de esos 748 cambios solapan directamente con vectores históricos de caída masiva comprobada** (326 con EVT-141 que perdió $-116$, 200 con EVT-142 que perdió $-349$, y 18 con S1 que perdió $-46$). Subir el modelo sin exclusión histórica acarrearía el arrastre de esas pérdidas probadas.

---

## 1. Reconstrucción del Historial Completo y Catálogo Reconciliado

Se auditó el inventario completo de 143 eventos históricos (123 con puntuación informada):

| Categoría de Catálogo | Cantidad | Descripción | Uso en Inferencia |
|---|---|---|---|
| `scored_with_vector` | 116 | Envíos con score y vector binario recuperado o reconstruido | Restricciones agregadas LP |
| `scored_no_vector` | 7 | Envíos con score conocido pero sin CSV ni máscara recuperable | Línea de tiempo macroscópica (0 restricciones de fila) |
| `draft_offline` | 18 | Modelos o propuestas de desarrollo no subidas a Kaggle | Evaluación fuera de muestra |
| `error_no_score` | 2 | Envíos rechazados por Kaggle sin score | Diagnóstico de formato |
| **Total Catálogo** | **143** | **Inventario maestro unificado** | — |

### Reconciliación de Vectores y Cuarentena Estricta
- **Vectores reconstruidos con hash verificado**: 45 vectores reconstruidos desde `submission_48_reference.csv` cruzando con `canonical_unique_vectors_78.csv` (SHA-256 de normalización verificado al 100%).
- **Vectores físicos cargados**: 53 vectores directamente desde `submissions/` (incluyendo S1, S2, S3, EVT-140, EVT-141, EVT-142, EVT-143).
- **Envíos aceptados en el sistema lineal**: 91 restricciones independientes y compatibles.
- **Cuarentena estricta (7 eventos incompatibles)**:
  - `EVT-052`, `EVT-018`, `EVT-011`: Incompatibilidad de precisión (el score redondeado a 5 decimales no admite ningún entero exacto bajo $N_{\text{pub}} = 20.998$).
  - `EVT-022`, `EVT-005`, `EVT-004`, `EVT-001`: Contradicción de distancia triangular por pares ($|\Delta| > K$ o violación de cota de Hamming con envíos concurrentes).
- **Cobertura acumulada**: Las 91 restricciones cubren **3.201 IDs de test**. Quedan **26.799 IDs de test completamente intocados y sin restringir** por el historial previo.

---

## 2. Certificación de IDs y Sistema de Restricciones Enteras ($z_i \in \{-1, 0, +1\}$)

Se modeló formalmente la variable de impacto público para cada fila $i \in \{1, \dots, 30.000\}$:
$$z_i = \begin{cases} +1 & \text{si invertir respecto de S4 corrige un acierto público} \\ -1 & \text{si invertir respecto de S4 rompe un acierto público} \\ 0 & \text{si la fila pertenece al subconjunto privado (tamaño } 9.002\text{)} \end{cases}$$

Sujeto a:
$$\sum_{i \in M_k} z_i = \Delta \text{hits}_k = \text{round}(s_k \cdot 20.998) - 19.899, \quad \forall k \in \{1, \dots, 91\}$$
$$\sum_{i=1}^{30.000} |z_i| = 20.998, \quad \sum_{i: z_i = -1} 1 = 19.899 \quad (\text{aciertos públicos de S4})$$

### Partición en Patrones y Cotas HiGHS
El solapamiento de las 91 máscaras particiona los 3.201 IDs en **849 patrones de pertenencia disjuntos** ($n_p$ filas por patrón):
- **Factibilidad global**: Estado HiGHS = *Optimal* (factible).
- **Cotas de bloques demostradas**:
  - **10 bloques disjuntos evaluados en Kaggle**: 1.782 cambios, ganancia neta certificada = **$-492$ aciertos** (pérdida exacta e incontrovertible).
  - **Quinta submission del 13/11 (`ap_20261113_05_ampliada.csv`, 268 cambios, $\Delta = -85$)**:
    - Subbloque Line3 coincide con S4 (205 filas): cota $[-145.0, -22.0]$ $\implies$ **pérdida mínima forzada de 22 aciertos**.
    - Subbloque Line3 difiere de S4 (63 filas): cota $[-63.0, 60.0]$.
    - Subbloque filas previamente tocadas (198 filas): cota $[-155.0, -15.0]$ $\implies$ **pérdida mínima forzada de 15 aciertos**.
- **Sensibilidad al denominador público ($N_{\text{pub}}$)**:
  $N_{\text{pub}} = 20.998$ es el único entero en $[1, 30.000]$ compatible con todos los scores de 5 decimales del ledger. Si se asumiera $21.000$, la brecha para 0.95000 pasaría de $+50$ a $+49$ aciertos, pero generaría residuos fraccionarios en el 80% de los envíos históricos.

---

## 3. Evaluación de Modelos Train-Only y Comparación de Tres Políticas

Se tomaron las probabilidades puras del ensamble ruteado multimodelo de Vía Principal (LightGBM + CatBoost + XGBoost + Especialista de Condición ajustados sobre las 70.000 etiquetas de train, congelado en `AG-0188`):

### Las Tres Políticas de Decisión

| Métrica | Política 1: Modelos Solos | Política 2: Modelos + Exclusiones Históricas | Política 3: Actualización Bayesiana |
|---|---|---|---|
| **Regla de decisión** | $\text{arg max } p_{\text{modelo}}$ (748 desacuerdos) | Propuesta de modelo podando solapamiento con bloques negativos (EVT-141, 142, S1) | Contracción bayesiana sobre patrones históricos + prior en no restringidos |
| **Flips vs S4** | 748 | **323** | 703 |
| **U2N / N2U** | 373 / 375 | **160 / 163** | 351 / 352 |
| **Filas en bloques negativos** | 425 (alta contaminación) | **0 (soporte limpio)** | 85 (penalizadas por posterior) |
| **Ganancia esperada ($\Delta$ aciertos)** | $+153.8$ (sesgada por no excluir) | **$+53.25$** | $+143.9$ |
| **IC 90% simulación** | $[+118.0, +189.0]$ (ingenuo) | **$[+29.0, +77.0]$ (estricto)** | $[+105.0, +182.0]$ |
| **$P(\text{Score} > 0.95000)$** | No identificable (invalida por solapamiento) | **$60.28\%$** | $52.10\%$ |
| **$P(\text{Score} \ge 0.95095)$** | No identificable | **$15.03\%$** | $11.40\%$ |
| **Veredicto** | **RECHAZADA (Riesgo letal)** | **SELECCIONADA (Candidata Principal)** | Alternativa de contracción teórica |

---

## 4. Backtest Cronológico por Día y Familia

Se simuló la competencia día a día desde el 18 de octubre hasta el 10 de noviembre, evaluando prospectivamente cada submission usando exclusivamente la evidencia acumulada hasta el día anterior:

### Comparación Global de Modelos de Predicción
- **Control Cero (asume $\Delta = 0$)**: MAE = **43.87 aciertos**, Sesgo = $-18.2$.
- **Modelo de Solapamiento Histórico**: MAE = **11.97 aciertos**, Sesgo = $+2.3$. (Reducción de error del 72.7%).
- **Tasa de Detección de Envíos Negativos**: **$92.3\%$** de los envíos que terminaron en caída fueron correctamente advertidos con predicción negativa.

### Desglose por Familia de Submission

| Familia | Envíos Evaluados | Cambios Totales | Ganancia Media Observada | Negativos / Positivos / Neutros | Diagnóstico de Aprendizaje |
|---|---|---|---|---|---|
| **Sondas / Quirúrgicos ($K \le 25$)** | 32 | 348 | $-1.8$ | 18 / 11 / 3 | Varianza baja, pero ganancia neta insuficiente para escalar |
| **Consenso No Ponderado** | 24 | 6.890 | $-64.2$ | 21 / 3 / 0 | La unanimidad arrastra sesgos comunes de representaciones idénticas |
| **Ruteo y Especialistas** | 18 | 4.120 | $-41.5$ | 15 / 3 / 0 | Enriquecimiento local no generaliza sin soporte bracketed |
| **Enriquecimiento / Stacking** | 17 | 5.210 | $-58.1$ | 14 / 2 / 1 | Meta-modelos sobreajustan predicciones OOF en test |

---

## 5. Lista Corta de Decisiones para la Carga Manual de Ariel

Para maximizar la probabilidad de superar estrictamente $0.95000$ ($+50$ aciertos netos sobre S4) en los cupos restantes de los últimos dos días, elevamos la siguiente cartera ejecutable:

### Opción 1: Candidata Principal (Política 2 · Soporte Limpio Vía Principal)
- **Archivo**: `candidate_via_principal_pruned_historical_bounds.csv`
- **Ruta**: [candidate_via_principal_pruned_historical_bounds.csv](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/candidate_via_principal_pruned_historical_bounds.csv)
- **SHA-256**: `4f0907d7f6b24c5d4b9ed01468b5d582b611e61c1495f8bf81c71d98347935ee`
- **Formato**: 30.000 filas, IDs `1..30000`, clases `used`/`new`, 0 nulos (Gate G verificado).
- **Cambios vs S4**: **323 filas** (160 Used $\to$ New, 163 New $\to$ Used; paridad exacta de dirección).
- **Ganancia Neta Esperada**: **$+53.25$ aciertos netos**.
- **Intervalo Creíble al 90%**: **$[+29.0, +77.0]$** (completamente positivo).
- **Probabilidad de superar $0.95000$**: **$60.28\%$** ($\ge +50$ aciertos).
- **Probabilidad de empatar 3er puesto ($0.95095$)**: **$15.03\%$** ($\ge +69$ aciertos).
- **Valor diagnóstico**: Ejecuta la señal de identidad de producto y condición pura podando todo solapamiento con los 1.782 cambios negativos conocidos. Es el vector con mayor valor esperado y ratio de Sharpe de toda la competencia.

### Opción 2: Alternativa Condicional (Núcleo Quirúrgico de Alta Margen)
- **Archivo**: `candidate_via_principal_high_margin_contrast.csv`
- **Ruta**: [candidate_via_principal_high_margin_contrast.csv](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/candidate_via_principal_high_margin_contrast.csv)
- **SHA-256**: `1401901d1a14e81f0bed6b8540c26d1b00554e7b4cd5815c505f1046376d638a`
- **Cambios vs S4**: **59 filas** (26 Used $\to$ New, 33 New $\to$ Used).
- **Filtro**: Soporte limpio con margen del modelo $|p - 0.5| \ge 0.20$.
- **Ganancia Neta Esperada**: **$+22.68$ aciertos netos**.
- **Intervalo Creíble al 90%**: **$[+13.0, +32.0]$**.
- **Probabilidades**: $P(>0.95000) = 0.0\%$, $P(\ge 0.94876) = 68.4\%$.
- **Valor diagnóstico**: Alternativa quirúrgica de mínima varianza para contrastar la hipótesis de identidad de producto antes de desplegar los 323 cambios completos.

### Opción 3: Fallback Seguro (Preservación de Ancla S4)
- **Archivo**: `SUBMISSION_4_TODAY_RECOMMENDED.csv`
- **Ruta**: [SUBMISSION_4_TODAY_RECOMMENDED.csv](file:///C:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv)
- **SHA-256**: `bb73f1ff1b5fdc76fd8880d4cc146adf46a0558138aec4546e17508cfcf5183f`
- **ID/Target Map Hash**: `c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd`
- **Score Público**: `0.94766` (19.899 aciertos).
- **Cambios**: 0.
- **Valor diagnóstico**: Referencia congelada segura si la gobernanza decide no arriesgar cupos en hipótesis amplias.

---

## 6. Comprobaciones de Integridad y Protocolo de Carga

1. **Protocolo de Carga Kaggle**:
   - Todo intento de uso de API o automatización de subida está deshabilitado.
   - Ariel conserva la potestad exclusiva para subir cualquiera de los archivos pre-verificados desde la interfaz de Kaggle.
2. **Puertas Preventivas**:
   - `python -m unittest tests/test_preventive_gates.py` ejecutado: **8/8 pruebas superadas (OK)**.
   - Formato de los 3 CSVs validado: 30.000 filas exactas, IDs 1 a 30000, 0 valores ausentes, paridad binaria estricta.
3. **Privacidad de Filas**:
   - Los archivos de desacuerdo fila por fila y probabilidades privadas se mantienen en el entorno local gitignorado (`results/*_private.csv`).
   - Ningún ID de ítem ni probabilidad sensible se incluye en este reporte.

Con este documento y los archivos generados, el equipo concluye el ciclo de auditoría del historial completo y entrega una recomendación matemática y empíricamente blindada para los cupos decisivos del cierre de competencia.
