# Reporte: Mezcla Predeclarada CatBoost Bag3 + NB-SVM (Corte Fresco 20261144)

- **Observaciones confirm**: `12264` (fracción vendedores vistos: `73.4%`)
- **Regla congelada en tune**: alpha_nbsvm = `0.05`, w_bag3 = `0.95`, umbral = `0.530`
- **Discrepancias entre Blend y Bag3**: `20` (0.16%)

## Rendimiento Pareado en Confirm

| Modelo / Configuración | Confirm Accuracy | Confirm Log Loss | Umbral Tune | Delta vs Single Histórico | Delta vs Bag3 | Bootstrap 95% CI (3.000 réplicas) |
|---|---|---|---|---|---|---|
| **CatBoost Single (Histórico)** | 0.95360 | 0.12290 | 0.485 | Control (0) | -21 | — |
| **CatBoost Bag3** | 0.95532 | 0.12035 | 0.515 | **+21** | Control (0) | `[5.0, 38.0]` vs Single |
| **NB-SVM Standalone** | 0.84915 | 0.34395 | 0.410 | **-1281** | **-1302** | — |
| **Blend Predeclarado (alpha=0.05)** | **0.95581** | **0.12419** | 0.530 | **+27** | **+6** | `[-3.0, 15.0]` vs Bag3 |

## Desglose Pareado por Vendedor (Visto vs Nuevo)

| Comparación | Fixes | Breaks | Net Total | Net Vendedores Vistos | Net Vendedores Nuevos |
|---|---|---|---|---|---|
| **Bag3 vs Single** | 45 | 24 | **+21** | +5 | +16 |
| **NB-SVM Standalone vs Bag3** | 212 | 1514 | **-1302** | -966 | -336 |
| **Blend Predeclarado vs Bag3** | 13 | 7 | **+6** | +3 | +3 |

## Archivos y Trazabilidad

- Pares por ID: `results\nbsvm_bag3_blend\paired_20261144.csv`
- Hash SHA-256 de pares: `10cd856ace7c5ab7b0f00dba37d0dfc1cdc2d0f74833ff1fe870aa7de9b77480`
- Reporte JSON: `results\nbsvm_bag3_blend\report_nbsvm_bag3_blend_20261144.json`
