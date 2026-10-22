# Auditoría de deep tabular + profesor NB

Estado al revisar: los checkpoints de profesor para ajuste están completos; el proceso original había quedado detenido en CatBoost de ajuste, sin `tune.npz` ni selección cerrada. Se avisó al coordinador. El coordinador reanuda con snapshots cada 30 segundos y dos hilos; no se lanzó entrenamiento duplicado desde esta auditoría.

## Comprobaciones favorables

- Los tres folds del profesor cubren las 49.000 filas de fit exactamente una vez, sin duplicados y sin ninguna fila de tune. Identidades de `train_rows` y `eval_rows` coinciden con la partición autoritativa.
- Cada vocabulario TF-IDF, log-count ratio NB y regresión logística se ajusta únicamente sobre el subconjunto de entrenamiento de su fold. La evaluación recibe un profesor ajustado sobre el entrenamiento completo de esa etapa.
- El nuevo profesor de dev se construye de nuevo con OOF dentro de 56.000 filas; confirm queda fuera. El refit final, si se realiza, usa solo las primeras 70.000 etiquetas.
- Las 101 features base no contienen columnas `condition`, `target` o `label`. `condition` se elimina antes de construirlas. Los atributos superiores cuyo id/nombre contiene `condition` se excluyen también antes del texto.
- Las estadísticas de precio por categoría/vendedor se calculan sobre el entrenamiento de la etapa y se mapean a evaluación, sin etiquetas. Que la fila training contribuya a su propia media de precio no es fuga de target.
- Árboles y umbral se eligen en tune; el lock se guarda antes de calcular confirm. La mezcla histórica 85/15 es fija y diagnóstica.
- El test conserva orden original, filas 70.000–99.999; el código no utiliza las etiquetas ficticias de test que se anexan como ceros al vector `y`.

## Límites y correcciones de interpretación

1. Las validaciones son heredadas y reutilizadas; no se debe llamar a confirm una validación virgen.
2. El profesor OOF ve dos tercios del entrenamiento mientras el profesor de evaluación ve todo; existe el cambio de distribución habitual al refitear un stacker.
3. Las cachés validan identidades de filas, pero no tienen hash de configuración y fuente. Al cambiar vocabulario, parámetros o datos, deben invalidarse; la huella de datos guardada por el coordinador debe conservarse con el checkpoint.
4. El coordinador añadió serialización atómica y recarga del profesor NB en las etapas confirm/full. Los mapas de precio aún se recomputan desde datos/cache; el `.cbm` aislado no es un scorer desplegable. El script auditor comprueba los archivos NB cuando aparecen.
5. `deep_variation_*` conserva solo agregados; no introduce directamente los valores de atributos de condición. Los ratios de stock/ventas son observacionales del snapshot, adecuados al problema de clasificación actual; no equivalen a variables disponibles antes de publicar.
6. La comparación histórica apropiada usa `refit49k_without_public` con umbral 0,505. La producción histórica es híbrida (base refit70k y algunos auxiliares congelados); no afirmar que todos sus componentes fueron entrenados con 70k.

El script `scripts/audit_deep_teacher_round3.py` regenera el JSON de comprobaciones y, cuando estén disponibles, compara confirm en las 4.147 filas comunes sin buscar pesos o umbrales nuevos.
