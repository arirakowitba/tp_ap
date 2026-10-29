# Ronda 142 — Validación externa honesta seller-group y evaluación de hipótesis complementarias

Fecha: 29-10-2026.
Código ejecutable: [`src/revision_submissions/seller_group_experiments_20261029.py`](../../src/revision_submissions/seller_group_experiments_20261029.py).
Resultados JSON: [`results/revision_submissions/round142_honest_validation_results.json`](round142_honest_validation_results.json).
Pares guardados: `paired_predictions_seed_2026.csv` y `paired_predictions_seed_314159.csv` (12.905 y 12.843 filas con IDs explícitos, etiquetas verdaderas y probabilidades de cada modelo).

## 1. Protocolo de partición y control de fuga

Se evaluaron dos semillas independientes (2026 y 314159) bajo un protocolo estricto seller-group:
- **Fit (~45k filas):** conjunto de entrenamiento del cual se derivan todos los transformadores supervisados.
- **Tune (~12k filas):** 10% de vendedores completamente nuevos (disjuntos de fit y confirm) + 10% de filas de vendedores vistos.
- **Confirm (~13k filas):** 10% de vendedores completamente nuevos (disjuntos de fit y tune) + 10% de filas de vendedores vistos.
- **Cero fuga:** Los vectorizadores TF-IDF (título palabras 1–2, caracteres 3–5, garantía y atributos), el codificador de metadatos, las medianas de precio por categoría y los modelos base se ajustaron **exclusivamente sobre fit**.
- Las probabilidades OOF del modelo lineal disperso sobre fit se generaron mediante un **GroupKFold de 4 folds agrupado por vendedor** dentro de fit, garantizando que ninguna fila de fit reciba una predicción de un modelo que vio a su propio vendedor.
- Los conjuntos confirm de ambas semillas comparten 2.337 IDs y 2.089 vendedores; no son réplicas independientes.

## 2. Rendimiento de modelos base en confirm

| Semilla | Modelo | Acc Total | Log Loss | Acc Vistos | Acc Nuevos |
|---:|---|---:|---:|---:|---:|
| 2026 | SparseLinear (TF-IDF + Meta) | 0.89609 | 0.26182 | 0.92314 | 0.88436 |
| 2026 | HistGB (+ OOF Sparse) | 0.90531 | 0.23724 | 0.93057 | 0.89436 |
| 2026 | LightGBM (+ OOF Sparse) | 0.90639 | 0.23418 | 0.93082 | 0.89580 |
| 2026 | **CatBoost** (+ OOF Sparse) | **0.90988** | **0.22834** | **0.93133** | **0.90058** |
| 314159 | SparseLinear (TF-IDF + Meta) | 0.88764 | 0.27649 | 0.92652 | 0.87107 |
| 314159 | HistGB (+ OOF Sparse) | 0.89340 | 0.25206 | 0.93121 | 0.87729 |
| 314159 | LightGBM (+ OOF Sparse) | 0.89504 | 0.24873 | 0.93174 | 0.87940 |
| 314159 | **CatBoost** (+ OOF Sparse) | **0.89940** | **0.24365** | **0.93200** | **0.88551** |

CatBoost supera de forma consistente a LightGBM, HistGB y al modelo lineal disperso en accuracy y log loss en ambos cortes.

## 3. Auditoría de la compuerta de historial de vendedor frente a baselines fuertes

Se aplicó la regla predeclarada ($n_{\text{fit}} \ge 3$, $|p_{\text{seller}} - 0.5| \ge 0.30$, y contradicción con el modelo base; combinación $0.6 p_{\text{seller}} + 0.4 p_{\text{base}}$):

| Semilla | Modelo Base | Filas Gated | Fixes / Breaks | Neto Confirm | IC 95% Bootstrap Vendedor |
|---:|---|---:|---:|---:|---:|
| 2026 | SparseLinear | 111 | 85 / 24 | +61 | [+34.0, +92.0] |
| 2026 | HistGB | 96 | 67 / 27 | +40 | [+17.0, +64.0] |
| 2026 | LightGBM | 94 | 65 / 26 | +39 | [+18.0, +61.0] |
| 2026 | CatBoost | 99 | 69 / 27 | +42 | [+20.0, +67.0] |
| 314159 | SparseLinear | 108 | 73 / 34 | +39 | [+13.0, +64.0] |
| 314159 | HistGB | 93 | 57 / 35 | +22 | [-1.0, +45.0] |
| 314159 | LightGBM | 91 | 56 / 34 | +22 | [0.0, +45.0] |
| 314159 | **CatBoost** | 78 | 43 / 33 | **+10** | **[-10.0, +29.0]** |

**Hallazgo crucial:** Frente a CatBoost en la semilla 314159, el neto cae a solo +10 y el **IC 95% bootstrap incluye valores negativos [-10, +29]**. Sumado a la pérdida neta de 47–49 aciertos informada en el holdout de ronda 13 contra modelos completos, queda confirmado empíricamente que la compuerta de historial de vendedor simple no es robusta contra modelos que ya integran señal contextual y no debe promoverse.

## 4. Hipótesis 1: Ensamble multi-modelo con calibración de umbral en tune

Se calibraron los pesos del ensamble convexo (CatBoost, LightGBM, SparseLinear) y el umbral de decisión en tune:
- Semilla 2026: pesos elegidos en tune `(0.6 Cat, 0.2 LGB, 0.2 Sparse)`, umbral `0.510`.
  - Confirm: Acc 0.90864 (vs 0.90988 CatBoost). Fixes: 74, Breaks: 90. **Neto: -16** (IC 95%: [-46.0, +11.0]).
- Semilla 314159: pesos elegidos en tune `(0.6 Cat, 0.4 LGB, 0.0 Sparse)`, umbral `0.525`.
  - Confirm: Acc 0.89901 (vs 0.89940 CatBoost). Fixes: 80, Breaks: 85. **Neto: -5** (IC 95%: [-38.0, +25.0]).

**Conclusión:** Un simple ensamble promedio ponderado con umbral afinado en tune no supera a CatBoost individual en confirm; la correlación de errores entre árboles de decisión hace que la calibración de umbral en tune sobreajuste ligeramente el ratio de clases del segmento nuevo.
