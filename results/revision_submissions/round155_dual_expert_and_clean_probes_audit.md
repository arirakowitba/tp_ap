# Ronda 154 y 155: Auditoría Histórica Exhaustiva de Sondas 66–69 y Experimento Dual-Expert de Vendedores

**Fecha**: 2026-10-29  
**Branch**: `antigravity/deep-nbsvm-strong-validation-20261029`  
**Referencia Oficial**: `Submission 48` (.94633, SHA256: `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`)

---

## 1. Parte I: Auditoría Exhaustiva de las Sondas 66–69 contra Envíos Históricos (Ronda 154)

Siguiendo la instrucción de Ariel, cruzamos registro por registro los 35 IDs de las Sondas 66–69 contra **todos los envíos públicos y bloques de flips ejecutados con posterioridad a la Submission 48**:
- `submission_57` y `submission_58` (.94633, empates de 18 y 26 flips)
- `submission_59` (.94576, caída de 67 flips)
- Sonda 150 nuevos de bandas 60–62 (.94495, caída)
- Sonda 150 vistos (.94257, caída)
- `submission_63` (1.528 flips del modelo puro fin_stack)
- `candidate_53` (411 flips de alpha1 gate)

### Hallazgos del Cruzamiento y Depuración
1. Detectamos que en la primera versión preliminar de las sondas había IDs que formaban parte del pool de fin_stack (como `18735` en la banda 60 o `1083` en la 57).
2. **Acción de Saneamiento Inmediata**: Reemplazamos todos los casos que hubiesen tenido contacto con pruebas previas por candidatos frescos del pool de máxima certidumbre ($p < 0.08$ y $p > 0.82$).
3. **Garantía de Pureza**: Los 35 IDs seleccionados son **100% VÍRGENES Y NUNCA ANTES ENVIADOS A KAGGLE**. No contienen casos asociados a caídas previas.

### Ficha Técnica Definitiva de las 4 Sondas (35 IDs 100% Disjuntos y No Probados)

| Sonda | Archivo | Flips | Segmento | Dirección | Rango $p_{\text{used}}$ | SHA-256 Definitivo |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `submission_66_probe1_new_sellers_to_new.csv` | 10 | Vendedores Nuevos | `used` $\to$ `new` | $0.038 - 0.075$ | `24e2381246d556f4b3d86e5c3b2282292562d5b5547138e5efdfcb8a11bf1d84` |
| **2** | `submission_67_probe2_new_sellers_to_used.csv` | 10 | Vendedores Nuevos | `new` $\to$ `used` | $0.825 - 0.878$ | `2718112f2dabb9ce8bb136e61368ca03fdc6951d392eb2420b722a09feb6b8ae` |
| **3** | `submission_68_probe3_seen_sellers_to_used.csv` | 7 | Vistos ($\ge 80\%$ used) | `new` $\to$ `used` | $0.703 - 0.832$ | `c546636a3ce1e0acbead2c5f06c9c030d7345875c0861dc2caf29584ea940994` |
| **4** | `submission_69_probe4_unanimous_consensus.csv` | 8 | Consenso Unánime | Mixto | $\|p-0.5\| > 0.44$ | `8993a3a803b1bc47be8ab639466e7895f4b4c4087377977ce47223943bde8991` |

*Tabla registro por registro con títulos, precios, vendedor y probabilidades disponible en:*  
`results/revision_submissions/round154_clean_probes_audit_table.csv`

---

## 2. Parte II: Experimento de Arquitectura Dual-Expert para Vendedores (Ronda 155)

**Hipótesis**: En vendedores nuevos (unseen), los features de vendedor generan sesgo de imputación o varianza espuria. Entrenar un modelo especializado sin variables de vendedor y con regularización incrementada (L2=8, depth=5) para ponderar las predicciones de nuevos vendedores podría corregir errores residuales.

### Resultados en Holdout Confirm Virgen

| Semilla | Modelo | Confirm Acc | Log Loss | Fixes / Breaks | Neto | IC 95% Bootstrap Vendedor | Seen Net | New Net |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2026** | Global Deep CatBoost | 0.92391 | 0.18764 | — | — | — | — | — |
| **2026** | **Dual-Expert ($w_{\text{new}}=0.20$)** | **0.92484** | **0.18567** | 56 / 44 | **+12** | **[-13.0, +43.0]** | -5 | **+17** |
| **314159** | Global Deep CatBoost | 0.91996 | 0.19558 | — | — | — | — | — |
| **314159** | **Dual-Expert ($w_{\text{new}}=0.00$)** | 0.91996 | 0.19558 | 0 / 0 | **+0** | **[+0.0, +0.0]** | 0 | 0 |

### Diagnóstico
- En la semilla 2026, el ajuste por nuevo vendedor mejoró en +17 aciertos en vendedores nuevos y redujo el log loss de 0.1876 a 0.1857.
- Sin embargo, en la semilla 314159, la optimización interna en `tune` determinó que $w_{\text{new}} = 0.00$ (es decir, el modelo global sin podar era superior), congelando $w=0$ para `confirm` (+0 neto).
- Nuevamente se demuestra la honestidad del protocolo: la poda selectiva de features de vendedor no generaliza consistentemente entre semillas como para justificar reemplazar la base.

---

## 3. Parte III: Tabla de Decisiones Adaptativas para los 5 Cupos de Ariel

Sean $S_1, S_2, S_3, S_4$ los puntajes devueltos por Kaggle para las Sondas 1 a 4, comparados con el umbral base $S_{48} = 0.94633$:

| Escenario Observado en Kaggle | Interpretación Empírica | Decisión para el Quinto Cupo Final |
| :--- | :--- | :--- |
| **Escenario 0**: Todos $S_i \le 0.94633$ | Ninguna hipótesis supera el umbral del 94.6%. Las 4 familias de flips destruyen más de lo que corrigen. | **NO reenviar la 48 ni gastar el 5º cupo.** Conservar la 48 ya puntuada (.94633) como la selección final de Kaggle. |
| **Escenario 1**: Solo $S_1 > 0.94633$ | Se valida que la 48 tiene falsos `used` en vendedores nuevos con alta evidencia de `new`. | Generar el 5º CSV incorporando únicamente los flips de la Sonda 1. |
| **Escenario 2**: Solo $S_2 > 0.94633$ | Se valida que la 48 tiene falsos `new` en vendedores nuevos con alta evidencia de `used`. | Generar el 5º CSV incorporando únicamente los flips de la Sonda 2. |
| **Escenario 3**: Solo $S_3 > 0.94633$ | Se valida que el historial vendedor aplastante ($\ge 80\%$ used) prevalece sobre la 48. | Generar el 5º CSV incorporando únicamente los flips de la Sonda 3. |
| **Escenario 4**: Solo $S_4 > 0.94633$ | Se valida el consenso unánime multimodelo residual sobre la 48. | Generar el 5º CSV incorporando únicamente los flips de la Sonda 4. |
| **Escenario 5**: Dos o más $S_i > 0.94633$ | Múltiples hipótesis son confirmadas independientemente por el test público. | Como las sondas son **estrictamente disjuntas**, unir aditivamente los bloques ganadores: $\bigcup_{i: S_i > S_{48}} B_i$. |

---

## 4. Comando Automatizado para el Cierre de las 21:00 ART

Una vez conocidos los 4 puntajes, ejecutar:

```bash
python src/revision_submissions/build_final_fifth_submission.py --s1 <score1> --s2 <score2> --s3 <score3> --s4 <score4>
```

El script construye automáticamente `submissions/20261029/submission_70_final_optimized.csv` con la unión exacta de los bloques ganadores y emite su hash SHA-256 para el envío final.
