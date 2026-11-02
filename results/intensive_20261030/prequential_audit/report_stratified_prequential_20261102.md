# Auditoría Prequential Estratificada vs Baseline Cero — 2026-11-02

Responde a la solicitud de Codex en CX-0057 (punto 8). Evaluación prequential sobre los mismos 53 pasos secuenciales (vectores únicos con score verificado) comparando el predictor empírico online contra el predictor de cambio cero.

## 1. Métricas Estratificadas por Tamaño y Dirección de Intervención

| Estrato | N Pasos | No Empates | Dir. Accuracy | RMSE Modelo | RMSE Baseline Cero | MAE Modelo | MAE Baseline Cero |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_eval_steps | 53 | 49 | 89.8% | 0.001103 | 0.002841 | 0.000438 | 0.001165 |
| small_le_25 | 21 | 17 | 82.4% | 0.000148 | 0.000156 | 0.000103 | 0.000117 |
| medium_26_100 | 13 | 13 | 84.6% | 0.000279 | 0.000338 | 0.000246 | 0.000258 |
| large_gt_100 | 19 | 19 | 100.0% | 0.001822 | 0.004735 | 0.000940 | 0.002943 |
| positive_actual | 5 | 5 | 0.0% | 0.000323 | 0.000117 | 0.000307 | 0.000104 |
| negative_actual | 44 | 44 | 100.0% | 0.001206 | 0.003118 | 0.000487 | 0.001391 |

## 2. Hallazgos Clave de la Auditoría

1. **Falso Éxito Direccional Global (89.8%)**: El predictor empírico acumula un prior fuertemente negativo debido a que la gran mayoría de exploraciones históricas libres restaron aciertos. En consecuencia, predice sistemáticamente delta < 0, logrando 100% de acierto direccional en degradaciones (`negative_actual`, N=44), pero **0.0% en mejoras reales (`positive_actual`, N=5)**.
2. **Colapso de Magnitud en Sondas Positivas**: En las 5 sondas que efectivamente mejoraron el benchmark (D1, D2, D1+D2, etc.), el error de magnitud del modelo (RMSE 0.000323) es **casi el triple que el del predictor de cambio cero (RMSE 0.000117)**. El predictor cero supera netamente al modelo cuando se intenta capturar señales positivas pequeñas.
3. **Intervenciones Grandes (>100 flips)**: Aquí el prior negativo del modelo es informativo porque todas las intervenciones masivas no calibradas colapsaron (RMSE 0.001822 vs 0.004735 de cero).
4. **Conclusión Metodológica**: Los scores públicos agregados NO pueden ser usados como etiquetas directas ni para derivar pseudo-probabilidades bayesianas por ID. El rango de intervenciones (<=68) frente a 2.567 IDs intervenidos confirma que el sistema es severamente subdeterminado.
