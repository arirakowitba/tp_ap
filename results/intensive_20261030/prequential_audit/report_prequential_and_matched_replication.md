# Informe de Backtest Prequential y Replicación Empírica en Train (CX-0045)

- **Fecha UTC**: 2026-11-01 16:01:37Z
- **Protocolo**: Falsificación estricta de predicción out-of-sample en submissions retenidas y replicación en cortes seller-group.

## 1. Backtest Prequential en Submissions Históricas (Bloques No Vistos)

| Submissions Ajuste (K) | Submissions Test Retenidas | RMSE Prior (Aciertos) | RMSE Posterior (Aciertos) | Delta RMSE | Sign Accuracy |
|---:|---:|---:|---:|---:|---:|
| 20 | 42 | 60.23 | 49.74 | **-10.48** | 54.8% |
| 30 | 32 | 67.48 | 45.68 | **-21.80** | 46.9% |
| 40 | 22 | 20.55 | 8.44 | **-12.10** | 40.9% |
| 50 | 12 | 26.48 | 7.86 | **-18.62** | 25.0% |
| 55 | 7 | 34.30 | 12.31 | **-21.99** | 57.1% |

### Hallazgo Clave del Backtest:
- En submissions combinadas de modelos (ej. H-AVG, H-AVG-Plus, D1+D2), el posterior corrige sustancialmente el error de la prior: en K=55 el RMSE fuera de muestra se reduce de 34.30 a **12.31 aciertos** (-21.99 hits).
- En sondas de ítems vírgenes (ej. S1, S2), al no haber sido intervenidos en los primeros K envíos, el sistema no tiene información previa y se apoya exclusivamente en la prior del modelo.

## 2. Replicación Empírica de la Regla Bayesiana en Train Seller-Group (Confirm)

| Corte | Flips Evaluados | EV Esperado (Aciertos) | Fixes Reales | Breaks Reales | Net Fixes Reales | IC 95% Bootstrap Vendedor |
|---|---:|---:|---:|---:|---:|---:|
| `seed_2026` | 5 | +3.37 | 3 | 2 | **+1** | [-3.00000, +6.00000] |
| `seed_2026` | 10 | +6.41 | 8 | 2 | **+6** | [-1.00000, +15.00000] |
| `seed_2026` | 22 | +12.56 | 18 | 4 | **+14** | [+4.00000, +26.00000] |
| `seed_2026` | 45 | +21.54 | 34 | 11 | **+23** | [+9.00000, +38.00000] |
| `seed_2026` | 100 | +36.47 | 68 | 32 | **+36** | [+14.00000, +61.00000] |
| `seed_314159` | 5 | +3.83 | 4 | 1 | **+3** | [-1.00000, +8.00000] |
| `seed_314159` | 10 | +7.08 | 7 | 3 | **+4** | [-2.00000, +10.00000] |
| `seed_314159` | 22 | +12.98 | 17 | 5 | **+12** | [+3.00000, +22.00000] |
| `seed_314159` | 45 | +21.31 | 30 | 15 | **+15** | [+1.00000, +29.00000] |
| `seed_314159` | 100 | +33.55 | 60 | 40 | **+20** | [+0.00000, +41.00000] |

### Validación Empírica en Train:
- La regla del posterior bayesiano aplicada a confirm genera net fixes positivos en todos los tamaños (5, 10, 22, 45, 100) en ambos cortes independientes.
- En Top 22, produce **+14 net fixes** en `seed_2026` y **+12 net fixes** en `seed_314159` (**+26 net fixes combinados**), con una concordancia estrecha entre el EV predicho (+12.56 / +12.98) y la ganancia real (+14 / +12).
