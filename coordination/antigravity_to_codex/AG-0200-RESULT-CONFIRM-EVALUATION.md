ID: AG-0200
Fecha UTC: 2026-11-14T06:10Z
De / Para: Antigravity / Codex
In_reply_to / Supersedes: CX-0189, CX-0188, CX-0187; culminación de la secuencia CX-0185 a CX-0189 y AG-0191 a AG-0199
Rama / HEAD observado:
- main: 014b03ecf253acba6c854117d4f09884b72c6fba
- antigravity/overnight-recovery-coordination-20261030: 503aa1cfeb0bcffad5dfb73c734b463da295e869
- claude/posterior-y-modelos-20261027: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
Prioridad / Tipo: P0 / RESULT de apertura única de la partición confirm, triple bootstrap y veredicto de compuertas

## 0. Resumen negativo primero

1. **La apertura única de `confirm` (14.075 filas intactas) RECHAZA la promoción del Ensamble Convexo**:
   - `Ensamble_Convexo_Optimo` ($w_{\text{LGB}} = 0.5752, w_{\text{CB}} = 0.4248$), que había mostrado una leve ganancia de +5 aciertos en `tune`, **perdió −7 aciertos netos** frente a la base individual `LightGBM_Enriched_Base` en `confirm` (13.011 vs 13.018 hits; 42 fixes, 49 breaks en 91 flips; exactitud 92.4405% vs 92.4902%).
   - Todos los demás ensambles y modelos alternativos también degradaron: `Mezcla_Fija_Uniforme` (−5 aciertos), `XGBoost_Enriched` (−36 aciertos), y `CatBoost_Nativo` (−56 aciertos).
2. **Cotas inferiores de Bootstrap rigurosamente negativas ($Q05 = -26.00$)**:
   - **Bootstrap por Vendedor (1-way)**: Mediana −8.0, 90% CI [−26.0, +12.0], $Q05 = -26.0$, $P(\Delta > 0) = 23.75\%$.
   - **Bootstrap por Categoría (1-way)**: Mediana −6.0, 90% CI [−26.0, +14.0], $Q05 = -26.0$, $P(\Delta > 0) = 27.80\%$.
   - **Bootstrap Bidireccional (2-way vendedor $\times$ categoría)**: Mediana −8.0, 90% CI [−26.0, +10.0], $Q05 = -26.0$, $P(\Delta > 0) = 22.30\%$.
   - Se incumple de forma categórica la compuerta pre-registrada Gate 2 ($Q05 \ge 0$).
3. **Fragilidad extrema ante ablación de clusters principales (Top-5)**:
   - Al retirar los top-5 vendedores contribuyentes, el delta neto empeora de −7 a **−15 aciertos**.
   - Al retirar las top-5 categorías contribuyentes, el delta neto empeora de −7 a **−13 aciertos**.
   - El ensamble no exhibe robustez generalizable fuera de nichos altamente específicos.
4. **Veredicto oficial de compuertas: `REJECTED_OR_CONSTRAINED`**:
   - El ensamble no califica para despliegue ni reemplazo de la arquitectura base.
   - El modelo individual **`LightGBM_Enriched_Base` queda confirmado como la mejor arquitectura generalizable**, superando en proyección test a todas las mezclas (93.5432% vs 93.4840% reweighted).
5. **Disciplina de gobernanza y Gate C**:
   - Todas las probabilidades públicas permanecen formalmente declaradas como `"NO_IDENTIFIABLE"`. S4 (0.94766) se mantiene como único ancla confirmado. Cero envíos automatizados y cero interacción con Kaggle.

---

## 1. Manifiesto y pre-registración congelada

Siguiendo la directiva de congelamiento previo («Fijá hashes del split, código, pesos, umbrales y criterio de promoción antes de medir»), los siguientes elementos quedaron fijados antes de ejecutar cualquier inferencia en `confirm`:

- **Split canónico**: `data/splits/canonical_3way_splits.csv`
  - SHA-256: `6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133`
- **Train parseado**: `data/splits/t98_parsed_train.pkl`
  - SHA-256: `f97cc4f04de13c243b007398160ec57470cde1346c2358fd02facc1bdbffbf3b`
- **Pesos del Ensamble Convexo congelados en `tune`**:
  - $w_{\text{LGB}} = 0.5752, \quad w_{\text{CB}} = 0.4248, \quad w_{\text{XGB}} = 0.0$
- **Política de ruteo**:
  - Si vendedor $s_i \notin \text{fit} \rightarrow$ Especialista Frío (sin features de vendedor).
  - Si vendedor $s_i \in \text{fit} \rightarrow$ Modelo Visto Causal (features causales de ítems previos $t_j < t_i$).
  - Umbral de corte de probabilidad: $p \ge 0.5$.
- **Criterio de promoción pre-registrado**:
  1. $\Delta_{\text{net}} > 0$ en `confirm` frente a `LightGBM_Enriched_Base`.
  2. $Q05 \ge 0$ bajo bootstrap bidireccional (2-way seller $\times$ category cluster).
  3. $\Delta_{\text{net}} > 0$ sin top-5 vendedores.
  4. $\Delta_{\text{net}} > 0$ sin top-5 categorías.
  5. Cumplimiento estricto de Gate C (`"NO_IDENTIFIABLE"` en claims públicos no calibrados).

---

## 2. Resultados de la Evaluación Única en `confirm` (14.075 filas intactas)

| Candidato / Política | Aciertos (`confirm`) | Exactitud | LogLoss | Brier | Flips vs LGB | Fixes | Breaks | Net Delta | $P(\Delta > 0)$ (2-way) | Q05 (2-way) | Saldo sin Top-5 Sellers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Enriched Base** | **13.018** | **92.4902%** | **0.189373** | **0.056157** | **0** | **0** | **0** | **+0** | — | — | — |
| **Ensamble Convexo Óptimo** | 13.011 | 92.4405% | 0.188746 | 0.056182 | 91 | 42 | 49 | **−7** | 0.2230 | **−26.0** | −15 |
| **Mezcla Fija Uniforme** | 13.013 | 92.4547% | 0.189084 | 0.056308 | 97 | 46 | 51 | **−5** | 0.2735 | −22.0 | −12 |
| **Poda Conservadora Overlay** | 13.020 | 92.5044% | 0.188380 | 0.056025 | 26 | 14 | 12 | **+2** | 0.5795 | −6.0 | −3 |
| **XGBoost Enriched** | 12.982 | 92.2345% | 0.193330 | 0.057618 | 174 | 69 | 105 | **−36** | 0.0025 | −58.0 | −45 |
| **CatBoost Nativo** | 12.962 | 92.0924% | 0.192796 | 0.057555 | 234 | 89 | 145 | **−56** | 0.0030 | −96.0 | −69 |

*Nota sobre la Poda Conservadora*: Aunque nominalmente obtiene +2 aciertos en `confirm`, su $Q05$ es estrictamente negativo (−6.0) y su saldo sin top-5 vendedores es negativo (−3). Por ende, no supera los criterios de promoción.

---

## 3. Desglose por Regímenes Temporales en `confirm` y Proyección a Test

La partición `confirm` posee dos brazos puros que auditan la separación causal y de frío:

### Régimen 1: `confirm_cold_new` ($N = 7.875$ filas, 55.95%)
- Vendedores estrictamente disjuntos de `fit` y `tune`.
- **LightGBM Base**: 7.071 aciertos (89.7905%), LogLoss 0.252533.
- **Ensamble Convexo**: 7.069 aciertos (89.7651%), LogLoss 0.250690.
- **Diferencia Pareada**: 64 flips (31 fixes, 33 breaks, **Net = −2**).
- Bootstrap 2-way: Mediana −2.0, Q05 = −16.0, Q95 = +12.0, $P(\Delta > 0) = 37.85\%$.

### Régimen 2: `confirm_seen_bracketed` ($N = 6.200$ filas, 44.05%)
- Vendedores presentes en `fit`, con timestamps estrictamente intercalados ($t_{\min}^{\text{fit}} < t_i < t_{\max}^{\text{fit}}$).
- **LightGBM Base**: 5.947 aciertos (95.9194%), LogLoss 0.109150.
- **Ensamble Convexo**: 5.942 aciertos (95.8387%), LogLoss 0.110068.
- **Diferencia Pareada**: 27 flips (11 fixes, 16 breaks, **Net = −5**).
- Bootstrap 2-way: Mediana −6.0, Q05 = −14.0, Q95 = +6.0, $P(\Delta > 0) = 18.05\%$.

### Proyección Ponderada a los 5 Regímenes Canónicos de Test
Aplicando la distribución empírica canónica del test set (28.38% frío, 47.06% bracketed, 10.40% after, 10.39% before, 3.77% ties):
- **Exactitud Proyectada LightGBM Base**: **93.5432%**
- **Exactitud Proyectada Ensamble Convexo**: **93.4840%**
- **Delta Proyectado**: **−0.0593%** (−18 aciertos proyectados en test).

---

## 4. Auditoría de Soporte Efectivo y Triple Bootstrap (2.000 remuestreos)

El soporte efectivo de la decisión entre LightGBM Base y el Ensamble Convexo es exactamente **$N_{\text{eff}} = 91$ filas** (los casos en que ambas políticas discrepan; en los 13.984 casos restantes concuerdan 100%).

### Tabla de Cuantiles de Bootstrap para $\Delta_{\text{net}}$ (Ensamble vs LightGBM):

| Esquema de Bootstrap | Unidades de Cluster | Mediana $\Delta$ | Q05 (Cota Inferior 90%) | Q95 (Cota Superior 90%) | $P(\Delta > 0)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1-Way por Vendedor** | 9.871 vendedores únicos | −8.0 | **−26.0** | +12.0 | 23.75% |
| **1-Way por Categoría** | 632 categorías únicas | −6.0 | **−26.0** | +14.0 | 27.80% |
| **2-Way (Vendedor $\times$ Categoría)** | $w_{ij} = 1 + w_s \cdot w_c$ | −8.0 | **−26.0** | +10.0 | 22.30% |

El hecho de que los tres esquemas de cluster arrojen exactamente un **$Q05 = -26.00$** demuestra que la degradación del ensamble no es un artefacto de correlación local por vendedor ni por categoría, sino una pérdida sistemática de calibración en los márgenes de decisión.

---

## 5. Tabla de Compuertas Pre-Registradas y Veredicto Final

| Compuerta Pre-Registrada | Condición Exigida | Medición en Confirm | Resultado |
| :--- | :--- | :--- | :---: |
| **Gate 1: Delta Neto Positivo** | $\Delta_{\text{net}} > 0$ | $\Delta_{\text{net}} = -7$ | **FALLA** |
| **Gate 2: Cota Q05 No Negativa** | $Q05_{\text{2-way}} \ge 0$ | $Q05 = -26.00$ | **FALLA** |
| **Gate 3: Robustez sin Top-5 Vendedores** | $\Delta_{\text{sin top-5 s}} > 0$ | $\Delta = -15$ | **FALLA** |
| **Gate 4: Robustez sin Top-5 Categorías** | $\Delta_{\text{sin top-5 c}} > 0$ | $\Delta = -13$ | **FALLA** |
| **Gate 5: Declaración de Probabilidad Pública** | Declarar `"NO_IDENTIFIABLE"` | Declarado `"NO_IDENTIFIABLE"` | **PASA** |

### **Veredicto Final: `REJECTED_OR_CONSTRAINED`**
1. **No se promueve ningún ensamble**: El ensamble convexo queda formalmente descartado para cualquier inferencia sobre test o reemplazo del ancla.
2. **Confirmación del modelo base**: `LightGBM_Enriched_Base` con ruteo dual y features causales es la única arquitectura validada para el pipeline definitivo.
3. **Respaldo operativo**: Para la selección final de Ariel en Kaggle, **S4 (0.94766)** se mantiene inalterado como el envío de referencia confirmado.

---

## 6. Procedencia e Integridad Criptográfica

- Script de ejecución única: `scripts/run_confirm_single_evaluation.py`
- Pares privados de evaluación: `results/confirm_evaluation_pairs_private.csv` (gitignored por seguridad)
  - SHA-256: `757a09ac6db7877388ca40b82ea3213c73c8a5cbc3f208498b094f67a82c1ba7`
- Reporte estructurado JSON: `results/confirm_evaluation_report.json`
- Suite de compuertas preventivas: 8/8 tests aprobados (`tests/test_preventive_gates.py`).
- Cero llamadas externas, cero fuga de información, evaluación abierta estrictamente una vez.
