# Cierre de tanda — 2026-10-22

Todos los entrenamientos programados terminaron, incluidos los refits completos y los dos ensambles finales. No se enviaron nuevas submissions.

Confirmación común de 14.000 filas (exploratoria por reutilización del holdout):

| Modelo | Accuracy | Log loss |
|---|---:|---:|
| Baseline nativo | 93.1571% | 0.169962 |
| CatBoost con teacher OOF | 94.2357% | 0.146576 |
| Seller dropout | 94.2500% | 0.148962 |
| Teachers separados título y garantía | 94.3643% | 0.145954 |
| Optimización TPE | 94.2071% | 0.146896 |
| Ensamble de cinco bases nuevas | 94.2786% | 0.145701 |

El modelo de campos mejora 169 aciertos netos frente al baseline en estas 14.000 filas; es el mayor accuracy observado de esta tanda, no prueba independiente de superioridad. El ensamble reduce levemente log loss pero no mejora accuracy frente a ese modelo. TPE mejoró tune sin confirmar mejora; no ampliar esa búsqueda por ese resultado solamente.

En las 4.147 filas comunes con el histórico: ensamble con histórico 94.3815% frente a 93.9715%, +17 aciertos netos (42 correcciones, 25 errores nuevos). Este subconjunto es distinto; no comparar su porcentaje directamente con la tabla de 14.000. Comparaciones exploratorias, múltiples modelos y holdouts reutilizados.

Ambos CSV finales tienen 30.000 predicciones y no incorporan flips. No tienen score público observado. Nuestro mejor score público observado sigue siendo 0.94533 (submission24). Submission26 conserva su condición de prueba pendiente; 0.94547 es una inferencia bajo supuestos, no un score medido.

Para retomar: conservar control público, probar de forma separada una base nueva y después diseñar flips sobre esa base. Los resultados agregados de cohortes antiguas no son etiquetas individuales ni se trasladan automáticamente a otra predicción. Decoder probabilístico validado contra fuerza bruta en problema pequeño, error máximo 3.33e-16; sus supuestos de independencia y tamaño público inferido siguen siendo limitaciones.

Fuentes: final/*/report.json, fields/confirm_report.json o reportes correspondientes, auditorías de esta ronda. Todos los refits nuevos usan 70.000 etiquetas de entrenamiento; histórico conserva auxiliares congelados. Caches regenerables y snapshots completados pueden omitirse del archivo; conservar scripts, probabilidades, modelos y teachers serializados.
