# Reporte Línea A: Modelo Lineal Textual de Alta Capacidad (Corte 20261143)

- **Dimensión del vocabulario sparse fit-only**: `171459` features
- **Observaciones confirm**: `12360` (fracción vendedores vistos: `72.8%`)
- **Discrepancias entre Línea A y CatBoost Bag3**: `1577` (12.76%)
- **Correlación de predicciones Línea A vs Bag3**: `0.8229`
- **Correlación de errores Línea A vs Bag3**: `0.3415`

## Resumen de Rendimiento Congelado en Confirm

| Modelo / Configuración | Confirm Accuracy | Confirm Log Loss | Umbral Tune | Delta vs Single Histórico | Delta vs Bag3 | Bootstrap 95% CI (3.000 réplicas) |
|---|---|---|---|---|---|---|
| **CatBoost Single (Histórico)** | 0.94709 | 0.13595 | 0.470 | Control (0) | -3 | — |
| **CatBoost Bag3** | 0.94733 | 0.13524 | 0.430 | **+3** | Control (0) | `[-15.0, 20.0]` |
| **Línea A Standalone** | 0.85647 | 0.33774 | 0.485 | **-1120** | **-1123** | — |
| **Blend Predeclarado (0.10 A + 0.90 Bag3)** | **0.94757** | **0.14265** | 0.505 | **+6** | **+3** | `[-14.0, 22.0]` vs Bag3 |

## Desglose Pareado por Tipo de Vendedor (Visto vs Nuevo)

| Comparación | Fixes | Breaks | Net Total | Net Vendedores Vistos | Net Vendedores Nuevos |
|---|---|---|---|---|---|
| **Bag3 vs Single** | 35 | 32 | **+3** | +4 | -1 |
| **Línea A Standalone vs Single** | 240 | 1360 | **-1120** | -851 | -269 |
| **Línea A Standalone vs Bag3** | 227 | 1350 | **-1123** | -855 | -268 |
| **Blend (0.10 A + 0.90 Bag3) vs Bag3** | 35 | 32 | **+3** | +4 | -1 |

## Archivos y Trazabilidad

- Reporte JSON: `results\line_a_text\report_line_a_text_20261143.json`
- Pares por ID: `results\line_a_text\paired_20261143.csv`
- Hashes SHA-256 de pares: `39e551da3db630a45d397bc1f729ddd00c10209cc5a3143984b557a197fb5783`
