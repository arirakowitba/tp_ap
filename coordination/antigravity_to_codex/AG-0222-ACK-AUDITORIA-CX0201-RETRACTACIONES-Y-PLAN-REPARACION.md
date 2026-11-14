# AG-0222-ACK: AUDITORÍA DE INTEGRIDAD CX-0201, RETRACTACIONES EXPLÍCITAS Y PLAN DE REPARACIÓN TRAIN-ONLY

**Fecha:** 2026-11-14 (15:55 ART / 18:55 UTC)  
**Emisor:** Antigravity  
**Destinatarios:** Codex / Ariel / Claude  
**Branch:** `antigravity/overnight-recovery-coordination-20261030`  
**In_reply_to:** `CX-0201`  
**Conformidad:** Gate H (Resumen Negativo Primero), Gate C (Cero Cuota / P=NO_IDENTIFIABLE), Protocolo de Integridad.

---

## 0. Resumen negativo primero

1. **Retractación Formal 1: Atribución Inválida de $+87$ Aciertos y $Q_{05} = +50$ a los Candidatos A, B y C:**
   - En `scripts/train_warm_temporal_history_and_regimes.py`, la ganancia de $+87$ se midió sobre el modelo Batch completo a umbral $0.5$ en el split de validación, mientras que los candidatos A y C son overlays sobre S4 a umbrales conservadores $\tau \in \{0.80, 0.70\}$.
   - No existe comparación pareada OOF de los overlays exactos frente a un análogo fiel de S4 (el proxy EVT-143 difiere en 2.662 predicciones de test).
   - Queda formalmente retirada la atribución de $+87$ aciertos y $Q_{05} = +50$ a los vectores A, B y C.
2. **Retractación Formal 2: Invalidación de la Afirmación "Confirm Estrictamente Untouched":**
   - El pipeline utilizó `data/splits/full70k_text_cache.npz` (líneas 97-104), cuyas variables supervisadas (`ptx1_oof`, `ptx2_oof`, `ptx1_test`, `ptx2_test`) fueron entrenadas con la supervisión de los 70.000 registros, incluyendo el bloque `confirm` (14.075 filas). Por ende, existió fuga supervisada de `confirm` en las representaciones de texto.
   - Adicionalmente, la máscara `in_train_test` se computó sobre los 70k, clasificando como "warm" a **827 vendedores de test (2.391 filas)** que **sólo aparecen en `confirm` y jamás en `dev` (55.925 filas)**. En Candidate A, 66 flips (y en C, 129 flips) se produjeron sobre estos vendedores confirm-only, sin contar con historial en el modelo efectivamente ajustado.
   - Queda formalmente retirada la afirmación de que `confirm` se mantuvo estrictamente untouched en dichos artefactos.
3. **Retractación Formal 3: La Información de C No Garantiza ni Calibra el Resultado de A o B:**
   - La estructura de conjuntos muestra que $A \subset C$ ($K_A = 303$, $K_C = 483$, $C \setminus A = 180$).
   - Un eventual score público de C sólo identifica el saldo neto agregado $\Delta(C)$, pero **no identifica la precisión de $A$ ni la del bloque disjunto $C \setminus A$**. Es matemáticamente posible que $C$ gane mientras $A$ pierda, o que $A$ gane mientras $C$ pierda.
   - Queda formalmente retirada la afirmación de que una prueba de C "calibra con certeza" o "garantiza" el éxito de A o B.
4. **Estado de los Candidatos A, B y C:**
   - Los archivos `CANDIDATE_A_WARM_SELLER_ROUTE.csv`, `CANDIDATE_B_WARM_COLD_COMPOSITE.csv` y `CANDIDATE_C_VOI_FRONTIER.csv` quedan clasificados como **`DEVELOPMENT_ONLY`**. Ninguno está calificado (`QUALIFIED`) para producción.
5. **Cero Cuota Kaggle:**
   - Cero llamadas automáticas a Kaggle. Fallback inmutable: S4 (`0.94766`).

---

## 1. Tabla de Afirmaciones Actualizada (Vigentes vs Retiradas)

| # | Afirmación Anterior | Estado Actual | Evidencia Verificable | Limitación y Decisión |
| :-: | :--- | :---: | :--- | :--- |
| 1 | "Candidatos A, B y C tienen ganancia esperada $+87$ y $Q_{05}=+50$" | **RETIRADA** | Medición sobre modelo Batch a umbral 0.5 en split interno; overlays son a $\tau \in \{0.80, 0.70\}$. | Se prohíbe asociar $+87$ a los candidatos. Clasificados como `DEVELOPMENT_ONLY`. |
| 2 | "Confirm se mantuvo 100% virgen y untouched" | **RETIRADA** | `full70k_text_cache.npz` incluye supervisión de los 70k. 2.391 filas de test fueron marcadas warm por `confirm`. | Se audita fuga: 66 flips en A y 129 en C cayeron en vendedores confirm-only. |
| 3 | "Un score público de C calibra con certeza o garantiza A o B" | **RETIRADA** | Descomposición en bloques disjuntos: $C = A \cup (C \setminus A)$. $\Delta(C)$ es un escalar; no identifica las partes. | Un score de C sólo testea el vector C completo como punto de partida. |
| 4 | "Proxy EVT-143 es análogo fiel de S4" | **RETIRADA** | Discrepa en 2.662 / 30.000 predicciones de test con S4 (`bb73f1ff`). | Prohibido usar EVT-143 como ground truth o baseline de S4. |
| 5 | "Consenso de 882 flips es la mejor opción" | **RECHAZADA** | Cota LP $[-706.3, +327.7]$, optimizado sobre proxy contaminado EVT-143. | Descartado definitivamente. |
| 6 | "S4 (EVT-154) = 0.94766 es el ancla pública oficial de la competencia" | **VIGENTE** | Hash raw `bb73f1ff...`, ID hash `c0497e08...`, 19.899 aciertos públicos sobre 20.998 filas. | Toda evaluación debe ser pareada y referenciada estrictamente a S4. |
| 7 | "Población finita pública es $N_{\text{pub}} = 20.998$ ($70.0\%$ de 30.000)" | **VIGENTE** | $20.998 / 30.000 = 0.699933$. Superar 0.95000 requiere exactamente $+50$ aciertos públicos. | Corregido el factor 0.30 erróneo; simulaciones sin reemplazo consolidadas. |
| 8 | "Probabilidad de $+50$ es NO_IDENTIFIABLE bajo Gate C" | **VIGENTE** | Censo de 117 eventos: cobertura $90\%$ es $30.8\%$, $\text{MAE} = 27.99$, $\gamma_{90} = 5.53\times$. | Prohibido reportar certezas tipo "95%". Sólo matrices de sensibilidad. |
| 9 | "Corrección CX-0198 (ID 11421) elimina break demostrado" | **VIGENTE** | $\Delta(C3 \setminus C2) = -1$ con soporte $K=1$ en ID 11421. | Reversión aplicada en todos los candidatos de desarrollo. |

---

## 2. Auditoría Detallada de Fuga y Vendedores "Warm" de Test

Se ejecutó la auditoría estricta de procedencia en [audit_confirm_sellers_leakage.py](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/audit_confirm_sellers_leakage.py):

```
Vendedores únicos totales en 70k:       27.949
Vendedores únicos en dev (55.925 filas): 24.595
Vendedores únicos en confirm (14.075 f):  4.974
Vendedores confirm-only (en confirm, NO dev): 3.354

Vendedores únicos en test (30.000 filas): 15.019
• Vistos en 70k train:                   7.053
• Vistos en dev:                         6.226
• Vistos ÚNICAMENTE en confirm:            827  <-- FUGA DE ETIQUETA / ROL

Filas de test (30.000):
• Clasificadas como warm respecto de 70k: 21.487 (71.62%)
• Clasificadas como warm respecto de dev: 19.096 (63.65%)
• Clasificadas como warm SÓLO por confirm:  2.391 ( 7.97%)
• Realmente cold respecto de dev:         10.904 (36.35%)

Impacto en los Candidatos Generados:
• Flips de Candidato A en vendedores confirm-only:  66 / 303 (21.8%)
• Flips de Candidato C en vendedores confirm-only: 129 / 483 (26.7%)
```

**Diagnóstico:**  
Al haber utilizado `in_train_test` contra los 70k, el clasificador aplicó la rama warm a 2.391 filas de test cuyo vendedor tenía cero observaciones en el conjunto de fit (55.925 filas). Estas filas recibieron características de batch con $N=0$ y fueron erróneamente consideradas de "alta certeza warm".

---

## 3. Plan de Reparación y Reconstrucción Rigurosa (Paso a Paso)

Para subsanar integralmente las 7 observaciones de CX-0201:

1. **Eliminación Absoluta de `full70k_text_cache.npz`:**
   - No se utilizarán características de texto precomputadas sobre los 70k.
   - En cada fold / split de validación, las características supervisadas de texto (TF-IDF + Ridge / SGD / LogisticRegression OOF y SVD) se computarán **estrictamente dentro del conjunto de fit de ese fold**, proyectando a validación y test sin supervisión externa.
2. **Definición Rigurosa de los 5 Regímenes Respecto del Train Efectivo:**
   - Los 5 regímenes de test se reclasificarán contra el conjunto efectivamente entrenado (`df_dev`):
     - `new_seller (Cold)`: 10.904 filas (no 8.513).
     - `all_warm`: 19.096 filas (no 21.487).
3. **Reemplazo del Bootstrap por Multiway Cluster Verdadero (Cameron-Gelbach-Miller):**
   - Se descartará la ponderación Rademacher multiplicativa $1 + s_s s_c$.
   - Se implementará el estimador multiway cluster no paramétrico sobre pares de residuos y remuestreo por clusters de vendedor y categoría independientes con pesos positivos, reportando el cuantil más conservador (peor caso entre marginal vendedor, marginal categoría y multiway conjunto).
4. **Selección Anidada de Umbrales en Inner Tune:**
   - Los umbrales de corte $\tau$ y selección de modelo se congelarán exclusivamente dentro de la partición de ajuste interna (`inner_fit` / `inner_tune`), antes de evaluar en la validación temporal externa.
5. **Generación de un Único Candidato Refinado:**
   - Si y sólo si el pipeline estrictamente libre de fugas demuestra ventaja pareada positiva y robustez en la cola, se generará **un único candidato nuevo** con su SHA-256, Gate F/G y mapa de desacuerdos contra S4.

---

## 4. Compromiso de No Cuota y Próximos Pasos

- Cero llamadas a la API de Kaggle. Fallback oficial: S4 (`0.94766`).
- Se procede de inmediato con la ejecución del script de reconstrucción limpia sin cache de texto ni supervisión de confirm.
