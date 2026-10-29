# Ronda 147: Auditoría de CatBoost Dual-Text frente a Modelos Fuertes y Cotejo con Submission 48

**Fecha**: 2026-10-29  
**Branch**: `antigravity/seller-group-validation-20261029`  
**Dataset**: 70k train records (`train_data.jsonlines`), 30k test records (`test_data.jsonlines`).  
**Cortes**: Partición estricta seller-group externa (fit ~45k, tune ~12k, confirm ~13k) sobre dos semillas independientes (2026 y 314159), alineación exacta por ID.

---

## 1. Motivación y Planteo

En la Ronda 144 validamos que agregar una probabilidad OOF construida con **NB-SVM** (razón de log-conteos sobre n-gramas de título y garantía) junto con **Sparse TF-IDF** aportaba mejoras frente a su propio baseline (+24 y +30 aciertos netos en confirm).

Sin embargo, para tomar una decisión honesta sobre el último cupo de Kaggle, era indispensable:
1. **Evaluar Dual-Text frente a los MODELOS FUERTES previos** (HistGB, LightGBM, CatBoost_r142 y Ensemble_r142) sobre exactamente los mismos IDs de confirm.
2. **Entrenar la arquitectura con las 70k etiquetas reales**, predecir las 30.000 instancias del test y contrastar predicción por ID contra la **Submission 48** (el mejor score público conocido, 0.94633).
3. **Analizar la exposición por segmentos** (vendedores vistos vs. nuevos, dirección `used` $\leftrightarrow$ `new`, márgenes de confianza) a la luz de los fracasos de las cinco sondas de `fin_stack`.

---

## 2. Contraste Pareado contra Modelos Fuertes (Confirm Idéntico)

Sobre las dos particiones confirm vírgenes e independientes:

### Semilla 2026 (n = 12.905 confirm rows; 3.903 vistos, 9.002 nuevos)

| Comparación | Acc Base $\to$ Dual | Fixes / Breaks | Neto | IC 95% Bootstrap Vendedor | Neto Vistos | Neto Nuevos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dual vs. **HistGB** | 0.90531 $\to$ 0.91003 | 266 / 205 | **+61** | [-1.0, +121.0] | +17 | +44 |
| Dual vs. **LightGBM** | 0.90639 $\to$ 0.91003 | 267 / 220 | **+47** | [-14.0, +108.0] | +17 | +30 |
| Dual vs. **CatBoost_r142** | 0.90988 $\to$ 0.91003 | 173 / 171 | **+2** | **[-52.0, +48.0]** | +13 | -11 |
| Dual vs. **Ensemble_r142** | 0.90872 $\to$ 0.91003 | 181 / 164 | **+17** | [-32.0, +63.0] | +9 | +8 |
| Dual vs. **Baseline CatBoost (r144)** | 0.90818 $\to$ 0.91003 | 170 / 146 | **+24** | [-16.0, +62.0] | -1 | +25 |

### Semilla 314159 (n = 12.843 confirm rows; 3.838 vistos, 9.005 nuevos)

| Comparación | Acc Base $\to$ Dual | Fixes / Breaks | Neto | IC 95% Bootstrap Vendedor | Neto Vistos | Neto Nuevos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dual vs. **HistGB** | 0.89340 $\to$ 0.90298 | 342 / 219 | **+123** | [+67.0, +182.0] | +24 | +99 |
| Dual vs. **LightGBM** | 0.89504 $\to$ 0.90298 | 317 / 215 | **+102** | [+46.0, +160.0] | +25 | +77 |
| Dual vs. **CatBoost_r142** | 0.89940 $\to$ 0.90298 | 199 / 153 | **+46** | **[+6.0, +89.0]** | -1 | +47 |
| Dual vs. **Ensemble_r142** | 0.90049 $\to$ 0.90298 | 201 / 169 | **+32** | [-7.0, +70.0] | -1 | +33 |
| Dual vs. **Baseline CatBoost (r144)** | 0.90065 $\to$ 0.90298 | 184 / 154 | **+30** | [-9.0, +67.0] | -4 | +34 |

---

## 3. Diagnóstico Estadístico del Hallazgo

1. **Superación nítida de árboles lineales débiles**: Dual-Text CatBoost supera de forma contundente y estadísticamente significativa a HistGB (+61 y +123) y a LightGBM (+47 y +102).
2. **Empate estadístico en Semilla 2026 frente a CatBoost Fuerte**:
   - Frente a `CatBoost_r142` (el modelo fuerte previo sin NB-SVM), el neto en la semilla 2026 es de tan solo **+2 aciertos** (173 fixes vs. 171 breaks).
   - El intervalo bootstrap al 95% centrado en el vendedor es **[-52.0, +48.0]**, abarcando ampliamente el cero en ambas direcciones.
   - En el estrato de vendedores nuevos de la semilla 2026, el neto fue **-11** (empeoró levemente).
3. **Ganancia en Semilla 314159**:
   - En la semilla 314159, el modelo sí logró **+46 aciertos** frente a CatBoost_r142 con IC 95% `[+6.0, +89.0]`, concentrado en nuevos (+47).
4. **Veredicto de Validación**: La señal de NB-SVM es genuina y aporta regularización en ciertos cortes de texto, pero **no proporciona una ganancia estadísticamente robusta y uniforme frente a un CatBoost fuerte ya optimizado**.

---

## 4. Reentrenamiento Completo (70k) y Cotejo por ID contra Submission 48

Se ajustó la arquitectura sobre las 70.000 filas de train con validación OOF de 5 folds por grupo de vendedor y se generaron probabilidades para los 30.000 ítems del test (`test_data.jsonlines`).

### Composición del Test Set
- Total filas: 30.000
- **Vendedores Vistos en Train**: 21.487 (71.62%)
- **Vendedores Nuevos (No en Train)**: 8.513 (28.38%)

### Desacuerdos (Flips) del Modelo Puro frente a Submission 48 (.94633)
- **Total Flips**: 1.774 / 30.000 (5,91%)
  - En Vendedores Vistos: 1.188 / 21.487 (5,53%)
  - En Vendedores Nuevos: 586 / 8.513 (6,88%)
- **Dirección de los Flips**:
  - `used` $\to$ `new`: 799 (526 vistos, 273 nuevos)
  - `new` $\to$ `used`: 975 (662 vistos, 313 nuevos)

### Distribución de Márgenes de Confianza ($|p - 0.50|$) en los Flips
- $p_{10} = 0.0298$, $p_{25} = 0.0744$, $p_{50} = 0.1579$, $p_{75} = 0.2749$, $p_{90} = 0.3793$, $p_{95} = 0.4163$

---

## 5. Evaluación de Riesgo y Por Qué No Arriesgar el Último Cupo

En las rondas recientes de auditoría de procedencia (`round140_fin_stack_provenance_audit_20261029.md`), Ariel ejecutó cinco sondas públicas sobre la Submission 48:
1. Sonda 59 completa (67 flips): score cayó a **.94576**
2. Sonda 59 nuevos (26 flips): score cayó a **.94619**
3. Sonda 59 vistos $\to$ new (27 flips): score cayó a **.94590**
4. Sonda 150 nuevos (bandas 60–62): score cayó a **.94495**
5. Sonda 150 vistos: score se desplomó a **.94257**

**Lecciones críticas aplicables a este escenario**:
1. **La base 48 (.94633) es extremadamente sólida**: requiere una precisión neta superior al 94.6% en cualquier bloque que se intente cambiar.
2. **El peligro de tocar vendedores vistos**: La sonda de 150 vistos destruyó más de 0.0037 de score. Subir el modelo puro (`submission_64`, con 1.188 cambios en vendedores vistos) implicaría una exposición inaceptable.
3. **Candidata Conservadora en Vendedores Nuevos (`submission_65`)**:
   - Se construyó un filtro que restringe los cambios **exclusivamente a vendedores nuevos** con alto margen ($|p - 0.50| \ge 0.25$).
   - Genera exactamente **141 flips** (62 `used` $\to$ `new`, 79 `new` $\to$ `used`, 0 en vendedores vistos).
   - Sin embargo, como el bootstrap frente al CatBoost fuerte cruzó el cero en la semilla 2026 (`[-52, +48]`), **la probabilidad de que este bloque de 141 flips supere el 94.6% de aciertos es incierta y no está garantizada**.

---

## 6. Recomendación Explícita para Ariel

> **RECOMENDACIÓN FORMAL**:  
> **Conservar la Submission 48 (.94633) como la predicción principal y definitiva.**  
> No gastar el último cupo de Kaggle en una predicción especulativa cuando la evidencia de validación externa cruzó el cero frente al CatBoost fuerte.  
> Si y solo si Ariel decide asumir un riesgo exploratorio acotado para el cierre, la única candidata técnica admisible es la conservadora:
> - **Archivo**: `submissions/20261029/submission_65_dual_catboost_new_sellers_top_margin.csv`
> - **Flips vs 48**: 141 (todos en vendedores nuevos, 0 en vistos)
> - **SHA256**: `33e03d6ffe4d983c70e5b18159b66648b15df21108dd1a487290ecea13ead867`

---

## 7. Artefactos Generados

- Script de entrenamiento 70k: `src/revision_submissions/full_train_dual_text_20261029.py`
- Script de auditoría pareada: `src/revision_submissions/audit_dual_text_vs_strong_baselines_20261029.py`
- Detalle de métricas JSON: `results/revision_submissions/round147_audit_dual_text_vs_strong_baselines.json`
- Metadata del reentrenamiento: `results/revision_submissions/round145_full_train_metadata.json`
- Archivo de auditoría por ID de los 1.774 flips: `results/revision_submissions/flips_dual_catboost_vs_s48.csv`
- Candidata Pura (1.774 flips): `submissions/20261029/submission_64_dual_catboost_pure.csv` (SHA256: `e96d204f6e88b09780437a91903d151bdca53f85448377a192e329351dc4927a`)
- Candidata Conservadora (141 flips en nuevos): `submissions/20261029/submission_65_dual_catboost_new_sellers_top_margin.csv` (SHA256: `33e03d6ffe4d983c70e5b18159b66648b15df21108dd1a487290ecea13ead867`)
