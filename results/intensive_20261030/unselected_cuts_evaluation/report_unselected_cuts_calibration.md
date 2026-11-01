# Informe de Validación en Cortes Seller-Group No Seleccionados y Calibración Extrema (CX-0043)

- **Fecha UTC**: 2026-11-01 13:43:12Z
- **Protocolo**: Selección de pesos y umbrales congelados estrictamente en `tune`; evaluación sin fugas en `confirm`.
- **Cortes Independientes Evaluados**: `seed=2026` y `seed=314159` (no seleccionados).

## 1. Resumen de Métricas Globales Pareadas en Confirm

| Corte | Filas Confirm | Peso Texto (Tune) | Umbral (Tune) | Acc Proxy | Acc Cand | Delta Acc | Net Fixes | IC 95% Bootstrap Vendedor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `seed_2026` | 12,905 | 0.00 | 0.470 | 0.92391 | 0.92391 | +0.00000 | +0 | [+0.00000, +0.00000] |
| `seed_314159` | 12,843 | 0.00 | 0.535 | 0.91996 | 0.91996 | +0.00000 | +0 | [+0.00000, +0.00000] |

## 2. Desglose Estratificado Vendedores Vistos vs Nuevos

| Corte | Estrato | Filas | Acc Proxy | Acc Cand | Fixes | Breaks | Net Fixes |
|---|---|---:|---:|---:|---:|---:|---:|
| `seed_2026` | Vistos | 3,903 | 0.95696 | 0.95696 | 0 | 0 | +0 |
| `seed_2026` | Nuevos | 9,002 | 0.90958 | 0.90958 | 0 | 0 | +0 |
| `seed_314159` | Vistos | 3,838 | 0.95102 | 0.95102 | 0 | 0 | +0 |
| `seed_314159` | Nuevos | 9,005 | 0.90672 | 0.90672 | 0 | 0 | +0 |
## 3. Calibración Empírica de Probabilidades Extremas

| Corte | Condición Extrema | Casos Confirm | Tasa Empírica Verdadera |
|---|---|---:|---:|
| `seed_2026` | P(used | p >= 0.90) | 4,327 | **97.41%** |
| `seed_2026` | P(new  | p <= 0.10) | 5,586 | **98.28%** |
| `seed_314159` | P(used | p >= 0.90) | 4,347 | **97.03%** |
| `seed_314159` | P(new  | p <= 0.10) | 5,352 | **98.54%** |

## 4. Política de Flips Clasificados por Valor Esperado (EV) en Confirm

| Corte | Flips Evaluados | Fixes | Breaks | Net Fixes |
|---|---|---:|---:|---:|
| `seed_2026` | top_5 (5 casos) | 5 | 0 | **+5** |
| `seed_2026` | top_10 (10 casos) | 8 | 2 | **+6** |
| `seed_2026` | top_20 (20 casos) | 14 | 6 | **+8** |
| `seed_2026` | top_22 (22 casos) | 14 | 8 | **+6** |
| `seed_2026` | top_30 (30 casos) | 20 | 10 | **+10** |
| `seed_2026` | top_50 (50 casos) | 29 | 21 | **+8** |
| `seed_314159` | top_5 (5 casos) | 1 | 4 | **-3** |
| `seed_314159` | top_10 (10 casos) | 4 | 6 | **-2** |
| `seed_314159` | top_20 (20 casos) | 11 | 9 | **+2** |
| `seed_314159` | top_22 (22 casos) | 13 | 9 | **+4** |
| `seed_314159` | top_30 (30 casos) | 19 | 11 | **+8** |
| `seed_314159` | top_50 (50 casos) | 29 | 21 | **+8** |
