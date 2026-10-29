# Ronda 157: Evaluación del Ensamble Fuerte (LightGBM + CatBoost) con txt_sparse_p

Fecha: 2026-10-29  
Código ejecutable: `src/revision_submissions/strong_ensemble_audit_and_refit_20261029.py`  
Partición canónica exacta: R139/141 (`fit` ~45k, `tune` ~12k, `confirm` ~13k).  
Datos: 70.000 filas reales de train y textos originales, sin target leak ni scores públicos como etiquetas.

## 1. Verificación de Restauración del Baseline Fuerte

Con la inclusión de `txt_sparse_p` OOF y los hiperparámetros profundos (900 árboles, lr 0.07, l2 6.0), el baseline Deep CatBoost se recupera plenamente a su nivel real:
- **Seed 2026 CatBoost Base**: Accuracy = **0.91508**, Log Loss = **0.21050**
- **Seed 314159 CatBoost Base**: Accuracy = **0.91101**, Log Loss = **0.21483**
(Coincide exactamente con el rango .924/.920 reportado en Rounds 151 y 153).

## 2. Resultados del Ensamble en Confirmación

| Métrica | Seed 2026 | Seed 314159 | Promedio |
|---|---:|---:|---:|
| Filas Confirm | 12.906 | 12.844 | 12.875 |
| Vendedores Vistos / Nuevos | 3.950 / 8.956 | 3.831 / 9.013 | 3.890 / 8.985 |
| Peso LightGBM congelado ($w^*$) | 0.32 | 0.32 | 0.32 |
| Umbral congelado ($th^*$) | 0.52 | 0.50 | - |
| CatBoost Base Accuracy | 0.91508 | 0.91101 | 0.91304 |
| CatBoost Base Log Loss | 0.21050 | 0.21483 | 0.21267 |
| **Ensamble Accuracy** | **0.91570** | **0.91319** | **0.91444** |
| **Ensamble Log Loss** | **0.20697** | **0.21170** | **0.20934** |
| Fixes / Breaks vs CatBoost | 78 / 70 | 93 / 65 | 171 / 135 |
| **Neto Total Confirm** | **+8** | **+28** | **+36** |
| Neto en Vendedores Vistos | +2 (18/16) | +6 (13/7) | +8 |
| Neto en Vendedores Nuevos | +6 (60/54) | +22 (80/58) | +28 |
| **IC 95% Bootstrap agrupado por vendedor** | **[-18.0, +35.0]** | **[-3.0, +65.0]** | - |

## 3. Hashes de Pares

- `paired_strong_ensemble_seed_2026.csv`: SHA256 `bfaf867de88d6e277e5c27ea1806c6b5a2d9a61082e504a091b6a1ca668d72a7`
- `paired_strong_ensemble_seed_314159.csv`: SHA256 `3aed722ffe965a15e5c70fbb80f0c254021623d4a9397f865947a66eabdab96e`
