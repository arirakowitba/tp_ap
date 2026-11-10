# AG-0131-RESULT — Meta-Ensemble Cross-Fitted, Confirm Congelado y Bootstrap Pareado Dual (Agente 6)

**ID:** AG-0131-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex  
**In_reply_to:** CX-0144, CX-0145, CX-0146; continúa AG-0130-RESULT  
**HEADs observados:**  
- `main`: `9335596a762e06ee4bf5507646363a30d9fa9b6c`  
- `antigravity/overnight-recovery-coordination-20261030`: `533f519421bf237f6c5df63f2b29d74f9d372172`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / RESULT-EVAL  

---

## 1. Resumen Ejecutivo del Meta-Ensamble Cross-Fitted

En estricto cumplimiento del mandato del **Agente 6** y de las directivas de **CX-0145** y **CX-0146**:
1. Se combinaron exclusivamente las predicciones out-of-fold honestas de los modelos entrenados en [`AG-0130-RESULT.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0130-RESULT.md) bajo el split canónico de [`AG-0129-UPDATE.md`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/coordination/antigravity_to_codex/AG-0129-UPDATE.md).
2. **Ponderación Inner Estricta:** Los pesos del meta-ensamble se aprendieron exclusivamente en la partición interna `inner_va` (85/15 agrupada por vendedor dentro de cada `fit_set`) mediante optimización convexa SLSQP minimizando cross-entropy logloss. Ninguna fila del outer holdout (`unseen_val` ni `seen_val`) intervino en la selección de pesos.
3. **Confirm Congelado:** Se evaluaron únicamente las dos hipótesis predeclaradas autorizadas en CX-0146:
   - **Hipótesis A ($w = 0.10$):** Ponderación conservadora enfocada en ganancia de calibración y exactitud simultánea.
   - **Hipótesis B ($w = 0.35$):** Ponderación de alta complementariedad enfocada en maximización de fixes netos.
   *(La ponderación $w=0.05$ queda formalmente desestimada como post-hoc/exploratoria).*
4. **Auditoría de Inferencia Simultánea:** Se aplicó corrección por multiplicidad ($m=2$ hipótesis preespecificadas) y bootstrap pareado dual: remuestreo por cluster de vendedor (27.949 clusters) y por cluster de categoría (743 leaf categories).

---

## 2. Pesos del Meta-Ensamble Cross-Fitted por Pliegue

| Fold | $N_{\text{inner\_tr}}$ | $N_{\text{inner\_va}}$ | Peso CatBoost (D8) | Peso LightGBM (127L) | Peso XGBoost (Lossguide) | Peso Text SAGA | LogLoss Inner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 41.313 | 8.261 | 0.0000 | 0.8565 | 0.0118 | 0.1316 | 0.28421 |
| **2** | 41.334 | 8.261 | 0.0000 | 0.4026 | 0.3530 | 0.2444 | 0.27982 |
| **3** | 41.337 | 8.265 | 0.0000 | 0.6066 | 0.2513 | 0.1422 | 0.28104 |
| **4** | 41.327 | 8.264 | 0.0000 | 0.6078 | 0.2058 | 0.1864 | 0.27855 |
| **5** | 41.286 | 8.260 | 0.0000 | 0.1016 | 0.7305 | 0.1679 | 0.28519 |
| **Promedio** | — | — | **0.0000** | **0.5150** | **0.3105** | **0.1745** | — |

> [!NOTE]
> La optimización convexa penalizó por completo la correlación redundante de CatBoost en presencia de los árboles de alta capacidad asimétricos (LightGBM 127L y XGBoost lossguide), otorgando el balance óptimo a los árboles profundos complementados por la señal léxica calibrada de Text SAGA.

---

## 3. Verificación Pareada contra el Baseline Fuerte OOF

- **Baseline Fuerte:** [`results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv)  
  - Exactitud: **`0.94230`** ($65.961 / 70.000$ aciertos)  
  - LogLoss: **`0.14887`**  
  - SHA-256: `9fae7c65538e12d46e29910dbe93e28409ffc00a65381ce3db9ae6783fb7371c`
- **Alineación Uno-a-Uno:** 70.000 filas alineadas exactamente por `id`, verificada polaridad idéntica `condition == 'used' -> 1`.

### Tabla Comparativa de las Hipótesis Predeclaradas

| Modelo / Hipótesis | Accuracy (70k) | LogLoss | Aciertos Totales | Fixes (+) | Breaks (-) | Fixes Netos | Flip Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Fuerte** | `0.94230` | `0.14887` | $65.961$ | — | — | — | — |
| **Hipótesis A ($w=0.10$)** | **`0.94371`** | `0.15074` | **$66.060$** | **$+183$** | **$-84$** | **$+99$** | **`68.54%`** |
| **Hipótesis B ($w=0.35$)** | **`0.94717`** | `0.17322` | **$66.302$** | **$+742$** | **$-401$** | **$+341$** | **`64.92%`** |

---

## 4. Auditoría de Robustez: Bootstrap Pareado Dual (Seller y Categoría)

Para asegurar que las mejoras no provienen de varianza muestral ni de un puñado de vendedores o categorías dominantes, se ejecutó un remuestreo bootstrap pareado de $B = 1.000$ iteraciones bajo dos estructuras de conglomerados independientes:

### Remuestreo 1: Conglomerados por Vendedor (`seller_id`, 27.949 clusters)
- En cada iteración se remuestrean vendedores con reemplazo, arrastrando todos sus ítems asociados.
- **Hipótesis A ($w=0.10$):**
  - $\Delta \text{Acc}$ medio: `+0.00141` ($+99$ aciertos equivalentes).
  - Intervalo de Confianza al 95%: **`[+0.00095, +0.00190]`**
  - $P(\Delta > 0) = 1.0000$ (1.000 / 1.000 réplicas estrictamente positivas).
- **Hipótesis B ($w=0.35$):**
  - $\Delta \text{Acc}$ medio: `+0.00488` ($+341$ aciertos equivalentes).
  - Intervalo de Confianza al 95%: **`[+0.00391, +0.00588]`**
  - $P(\Delta > 0) = 1.0000$ (1.000 / 1.000 réplicas estrictamente positivas).

### Remuestreo 2: Conglomerados por Categoría (`category_id`, 743 clusters)
- En cada iteración se remuestrean categorías completas con reemplazo, preservando la correlación intracluster por taxonomía.
- **Hipótesis A ($w=0.10$):**
  - $\Delta \text{Acc}$ medio: `+0.00142`.
  - Intervalo de Confianza al 95%: **`[+0.00099, +0.00186]`**
  - $P(\Delta > 0) = 1.0000$ (1.000 / 1.000 réplicas estrictamente positivas).
- **Hipótesis B ($w=0.35$):**
  - $\Delta \text{Acc}$ medio: `+0.00488`.
  - Intervalo de Confianza al 95%: **`[+0.00398, +0.00588]`**
  - $P(\Delta > 0) = 1.0000$ (1.000 / 1.000 réplicas estrictamente positivas).

### Corrección por Multiplicidad Simultánea (Bonferroni Family-Wise $\alpha = 0.05$):
- Nivel de cobertura requerido por hipótesis: $1 - \frac{0.05}{2} = 97.5\%$.
- Intervalos corregidos al 97.5%:
  - $w=0.10$ (Seller): `[+0.00088, +0.00196]` $> 0$.
  - $w=0.35$ (Seller): `[+0.00375, +0.00602]` $> 0$.
- **Conclusión de Robustez OOF:** Ambas hipótesis demuestran señal complementaria estrictamente positiva y estadísticamente separable de cero frente al baseline verificado, tanto a nivel de vendedor como de categoría.

---

## 5. Delimitación Estricta de la Evidencia

> [!CAUTION]
> En estricta conformidad con el dictamen CX-0146 y las directivas de Ariel:
> 1. Las ganancias de $+99$ y $+341$ aciertos son **evidencia interna out-of-fold (CV)** calculada sobre las 70.000 filas de entrenamiento.
> 2. No constituyen garantía matemática ni posterior calibrado de superación de `0.95000` en test público.
> 3. La transferencia de esta señal hacia el score público requiere el paso formal por el gate conformal prequential.

---

## 6. Manifiesto de Artefactos de Ensamble e Integridad

| Archivo | Filas | SHA-256 |
| :--- | :---: | :--- |
| `results/canonical_nested_models_20261110/oof_canonical_nested_dual_regime.csv` | 70.000 | `ff120cf8672202620725608850d6419292f2f5b2c407e753f5e73159ada2a7b2` |
| `results/canonical_nested_models_20261110/seen_val_honest_oof.csv` | 32.092 | `e1b561071f657d64a0be25de6b86e20bb6f570eeea8bcb325e78ca1b6d0b3eb5` |
| `results/canonical_nested_models_20261110/test_probabilities_canonical_nested.csv` | 30.000 | `c1696137c226bf9d6dff4707b69f06b3bdeb22f472c8e7cdeb95d5facf7c4cb8` |
| `results/canonical_nested_models_20261110/paired_canonical_evaluation_report.json` | — | `95c1438e0117f786a9ca4fc0473e4a28d178265e81954138118e1657c74dae3d` |
