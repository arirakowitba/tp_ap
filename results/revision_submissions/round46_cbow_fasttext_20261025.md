# Ronda 46: FastText CBOW adicional

Se entrenó FastText CBOW de 40 dimensiones, ventana 8 y 10 épocas, sin etiquetas, sobre los 100.000 textos originales. Sus vectores se agregaron al FastText skip-gram existente y se volvió a entrenar la segunda etapa HistGB en dos cortes disjuntos por vendedor. Comparación pareada contra HistGB + FastText skip-gram, umbral .475.

| Corte | Tune fixes/breaks/net | Confirm fixes/breaks/net | Log loss confirm, base → CBOW |
| --- | ---: | ---: | ---: |
| 1 | 79/73/+6 | 77/77/0 | .210628 → .210559 |
| 2 | 99/88/+11 | 92/101/−9 | .205913 → .206322 |

La mejora de tune no se reprodujo en confirm, y el segundo confirm empeoró tanto en aciertos como en log loss. No se promueve como candidata. Ambos cortes provienen del mismo train; no equivalen a evaluación pública ni reproducen la política exacta de la submission 33. Los NPZ contienen probabilidades e índices pareados para análisis posterior. No se utilizaron etiquetas de test ni se envió submission.
