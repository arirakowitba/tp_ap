# Corrección de auditoría de round54 (2026-10-26)

El reporte inicial de round54 dijo que la comparación contra CatBoost usaba umbral .5. La implementación `round54_fresh_cat.py` aplicó .475 a CatBoost. Se reabrieron las probabilidades pareadas guardadas y se recalcularon ambos umbrales para el subconjunto FastText `used→new` (119 filas en cada sección):

| Sección | FastText vs HistGB (.475) | FastText vs CatBoost (.475) | FastText vs CatBoost (.5) |
|---|---:|---:|---:|
| Tune | −3 | +1 | −3 |
| Confirm | +15 | +3 | +3 |

La conclusión sustantiva sigue siendo no promover: confirm es apenas +3 frente a CatBoost y tune no lo respalda con el umbral .5. Los cortes previos que utilizaron CatBoost .5 no deben mezclarse con la columna CatBoost .475. Esta corrección prevalece sobre el texto de round54.

Auditoría de margen preexistente en round54: con `p_FastText <= .425`, hubo 63 cambios en tune y 72 en confirm; FastText vs CatBoost .5 obtuvo −7 y +2, respectivamente. El margen mayor tampoco estabiliza la señal. No se usaron etiquetas de test ni cuota de Kaggle.
