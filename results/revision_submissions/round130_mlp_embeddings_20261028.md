# Ronda 130 — red pequeña sobre embeddings y cantidades

MLP 64 y 64×32 unidades con regularización alpha=10 y early stopping, entrada FastText de textos originales más log(precio) y log(cantidad). Etiquetas solo train, fit vendedor-disjunto del holdout; tune y confirm disjuntos dentro de cada corte. Baseline pareado HistGB+FastText ronda39. Mezclas .05,.10,.20,.35; regla predeclarada: tune exige neto de aciertos >0 y log loss menor.

En los tres cortes a/b/c ninguna mezcla pasó ambos criterios de tune; se seleccionó la base en todos. No hay ganancia confirmada ni predicción de test que promover. Los confirms a/b/c solapan 2868, 3218 y 3185 filas por pareja, 616 en los tres, por lo que no son réplicas independientes. No hay pares CatBoost en esos mismos cortes. Test sin etiquetas; scores públicos no se usaron en el entrenamiento.

Checkpoint con script, reportes y pares completos: Library `libfile_a009c0529e9c8191aa57649b1014f6f5`, SHA256 `2fb47c6770dfcbaaeef2f180ea3bc218a7c3b655909d8a495263a3f0974ce5ba`.
