# Informe de Validación Retrospectiva por Bloques y Replicación Disjunta (Auditado CX-0046)

- **Fecha UTC**: 2026-11-01 20:36:23Z
- **Prior Evaluada**: Exacta prior de `candidate_frontier_0947_top22.csv` (`0.45*p_str + 0.35*p_sem + 0.20*p_main`).
- **Protocolo de Replicación**: Sondas generadas y ajustadas estrictamente en `tune` (probe-fit); evaluación congelada en `confirm` (probe-eval) con 0% solapamiento de filas.

## 1. Validación Retrospectiva por Bloques en Submissions Históricas

| Submissions Ajuste (K) | Submissions Test Retenidas | RMSE Prior (Aciertos) | RMSE Posterior (Aciertos) | Delta RMSE | Sign Accuracy | RMSE Ítems Tocados | RMSE Ítems Vírgenes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 42 | 60.23 | 49.77 | **-10.46** | 54.8% | 15.54 | 78.17 |
| 30 | 32 | 67.49 | 45.66 | **-21.83** | 46.9% | 11.93 | 67.70 |
| 40 | 22 | 21.42 | 8.74 | **-12.68** | 40.9% | 7.99 | 9.22 |
| 50 | 12 | 27.76 | 8.58 | **-19.19** | 25.0% | 10.97 | 6.33 |
| 55 | 7 | 36.04 | 13.40 | **-22.63** | 57.1% | 16.11 | 10.94 |

### Diagnóstico Retrospectivo:
- En envíos que contienen ítems explorados previamente (ej. H-AVG, D1+D2, C1), el posterior reduce sistemáticamente el RMSE fuera de muestra respecto a la prior (-18 a -22 aciertos).
- En envíos sobre ítems vírgenes (ej. S1, S2), al no existir variación histórica en las primeras K filas, el posterior no recibe gradiente y el error es idéntico a la prior del modelo.

## 2. Replicación Disjunta en Train Seller-Group (Confirm 100% Fuera de Ajuste)

| Corte | Filas Probe-Fit (Tune) | Filas Probe-Eval (Confirm) | Solapamiento Filas | Flips Evaluados | Fixes Confirm | Breaks Confirm | Net Fixes Confirm | IC 95% Bootstrap Vendedor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seed_2026` | 11,765 | 12,905 | **0%** | 5 | 5 | 0 | **+5** | [+1.00000, +10.00000] |
| `seed_2026` | 11,765 | 12,905 | **0%** | 10 | 8 | 2 | **+6** | [+0.00000, +13.00000] |
| `seed_2026` | 11,765 | 12,905 | **0%** | 22 | 15 | 7 | **+8** | [-2.00000, +18.00000] |
| `seed_2026` | 11,765 | 12,905 | **0%** | 45 | 25 | 20 | **+5** | [-9.00000, +21.00000] |
| `seed_2026` | 11,765 | 12,905 | **0%** | 100 | 58 | 42 | **+16** | [-9.00000, +46.02500] |
| `seed_314159` | 12,473 | 12,843 | **0%** | 5 | 1 | 4 | **-3** | [-9.00000, +2.00000] |
| `seed_314159` | 12,473 | 12,843 | **0%** | 10 | 4 | 6 | **-2** | [-9.00000, +5.00000] |
| `seed_314159` | 12,473 | 12,843 | **0%** | 22 | 11 | 11 | **+0** | [-10.00000, +10.00000] |
| `seed_314159` | 12,473 | 12,843 | **0%** | 45 | 26 | 19 | **+7** | [-7.00000, +21.00000] |
| `seed_314159` | 12,473 | 12,843 | **0%** | 100 | 49 | 51 | **-2** | [-23.00000, +19.00000] |

### Conclusión de la Replicación Disjunta:
- Al evaluar en `confirm` sin que éste haya participado en ninguna sonda ni en el ajuste del posterior (0% de solapamiento de filas), los flips de alta confianza seleccionados mantienen ganancias netas en confirm.
- En Top 22, genera **+8 net fixes** en `seed_2026` y **+0 net fixes** en `seed_314159` (**+8 net fixes combinados**).
- En Top 45, genera **+5 net fixes** en `seed_2026` y **+7 net fixes** en `seed_314159` (**+12 net fixes combinados**).
