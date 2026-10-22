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

## Resultado de confirmación reanudada

Se cerró selección en tune antes de confirmar: 749 árboles y umbral 0,51 (accuracy de tune 93,6857%). El profesor de dev56k pasó las mismas comprobaciones de exclusión, cobertura, identidad y coincidencia numérica OOF. Su archivo joblib se recargó correctamente.

En confirm14k el candidato obtuvo 94,2357%, logloss0,146576, vistos96,3283% y nuevos89,4786%. Mejora al CatBoost nativo de referencia, pero ese baseline es más débil que la composición histórica.

En las4.147 filas comunes mutuamente excluidas, el histórico `without_public` a0,505 obtuvo93,9715%; el candidato a0,51 obtuvo94,2850%:70correcciones y57errores nuevos, saldo+13, p pareado noajustado≈0,287. Logloss pasó de0,155061 a0,146291. La mezcla fija85%histórico/15%candidato obtuvo94,0439%.

La mejora de accuracy frente al histórico es pequeña y no concluyente; no se buscó un peso nuevo sobre estas filas. El `.cbm` de confirmación recargó correctamente con749árboles. El refit completo y sus predicciones de test estaban en marcha al actualizar esta sección.

## Cierre del refit completo

El refit completo terminó correctamente. El profesor completo cubre70.000 filas OOF sin duplicados y predice únicamente las30.000 filas de test. El profesor joblib de confirm y el de full recargan, sus filas de entrenamiento y anchuras de vocabulario/ratio/modelo coinciden, y conservan clases[0,1]. El modelo CatBoost final recarga con749árboles y el vector test contiene30.000 probabilidades finitas dentro de[0,1]. No se observaron cambios de mapping, identidad o fuga entre etapas. Los mapas de contexto aún se recomputan desde raw/cache.
