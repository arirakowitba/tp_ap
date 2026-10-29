# Estado público y dirección de entrenamiento — 28 de octubre de 2026

## Snapshot informado por Ariel

| Puesto | Equipo | Score | Entradas |
|---:|---|---:|---:|
| 1 | Maximo Lladhon | 0.95404 | 11 |
| 2 | Nicolás Caveda | 0.95095 | 15 |
| 3 | Gonzalo Mariano Bühler | 0.94895 | 22 |
| 4 | Clara Rodríguez Acevedo | 0.94747 | 3 |
| 5 | Ezequiel Naymark | 0.94690 | 11 |
| 6 | Nazareno Ohanian | 0.94680 | 9 |
| 7 | Agustín Elías Sánchez | 0.94671 | 5 |
| 8 | Valentin Zuppa | 0.94638 | 17 |
| 9 | Ariel Rakowszczyk | 0.94633 | 81 |

La última submission de Ariel, muestra de 150 vendedores vistos de `fin_stack`, fue 0.94257. La mejor sigue siendo 48=.94633. Brecha al líder .00771 (≈162 aciertos si hay 21.000 públicos). El denominador público y los scores redondeados hacen aproximadas las conversiones a aciertos. Fuente: snapshot comunicado por Ariel. Los cinco CSV, hashes, tamaños y relaciones de bloque están en [round138_fin_stack_segment_samples_20261028.md](round138_fin_stack_segment_samples_20261028.md). No tratar ranking, conteo de entries ni scores como etiquetas por ID.

## Interpretación del fracaso de `fin_stack`

La CV estratificada aleatoria .9548/.1191 no es prueba de traslado a test ni validación de política 48. El stack cambió 67 IDs de máxima confianza en 59 y perdió ≈12; sus muestras independientes adicionales de 150 nuevos y 150 vistos perdieron ≈29 y ≈79. La muestra vista contiene 132 cambios →new, de modo que no separa un efecto de vendedor de uno de dirección. No extrapolar saldo por ID. Las cinco pruebas públicas no se suman porque 59 contiene sus dos particiones. No expandir 60–63 ni usar posterior público para entrenar.

## Instrucción de continuación para Claude y para este proyecto

1. Partir de main posterior a `c781798`, integrar estos cinco scores como **restricciones de bloques**; la rama `claude/posterior-y-modelos-20261027` sigue en `37e6106` y no aporta commits nuevos respecto de la revisión anterior.
2. Reconstruir cada feature de texto, TE, sellerother y estadística de vendedor **dentro de cada fit**. Para meta OOF, generar bases internas que no vean ni la etiqueta propia ni la del confirm externo. Auditar sus índices y fecha de selección de hiperparámetros. El `stk_sellerother` original filtraba y; quitar esa columna no prueba por sí solo ausencia de dependencias entre modelos base y meta folds.
3. Comparar en los **mismos** confirm seller-group, con tune congelado: política proxy de 48 calculable sólo con train, HistGB+FastText, CatBoost sin sellerother, XGBoost y stack. Informar aciertos, fixes/breaks pareados, log loss, seen/new y por dirección; bootstrap agrupado por vendedor e intervalos. Auditar solapamiento de vendedores entre los cortes previos. No elegir confirm por el resultado.
4. Hipótesis prioritaria: por qué el stack cambia tantos vistos→new y falla. Medir distribución de conteo de vendedor, condición mayoritaria del fit, probabilidades y error de train etiquetado; hacer ablation de TE, seller_id y precio relativo en folds externos. Si la ganancia sólo existe en CV aleatoria, reportarla como tal.
5. Explorar una nueva base con información complementaria y sin datos externos etiquetados; preferir selección de iteraciones e hiperparámetros en tune interno. Conservar predicciones pareadas por ID, semillas, config, hashes de entradas y manifiesto de filas en checkpoint. Promover un CSV de test sólo tras confirm no seleccionado o como prueba explícitamente diagnóstica. Ariel controla las submissions.

La caída pública no implica que todo el trabajo de Claude sea inútil: los modelos pueden aportar en un segmento o como calibración. Hace falta distinguir ese aporte con etiquetas reales y una validación que reproduzca la mezcla seen/new del test.

## Objetivo de la próxima jornada (pedido de Ariel)

Preparar una **última submission de explotación** para la noche del 29-10 que maximice la estimación de score y ayude a recuperar posiciones. No prometer que supere 48=.94633: los cinco resultados recientes mostraron que la aparente confianza del stack podía ser contraproducente. Distinguir pruebas diagnósticas tempranas de la decisión final, sin gastar cupos desde este proceso.

Plan secuencial y condición de promoción:

1. Antes de cualquier CSV nuevo, ejecutar validación etiquetada de baselines y candidatos sobre los **mismos IDs**, con fit/tune/confirm externos y seller-group; registrar para visto/nuevo accuracy, logloss, fixes/breaks, IC bootstrap vendedor y solapamiento de cortes. Repetir el mecanismo de historial de vendedor frente a modelo fuerte, no frente al título débil.
2. Auditar el stack de Claude: TE, sellerother, dependencia OOF entre folds, selección de iteraciones y pesos negativos; ejecutar ablations en confirm externo sin selección retrospectiva. Si no hay ventaja robusta, no usar `fin_stack` como predictor de flips.
3. Si aparece una mejora estable en confirm que también supera una proxy razonable de política 48, entrenar en 70k y generar predicción completa de test con manifiesto por ID y SHA256. Comparar diferencias con la 48 por segmento, dirección, conteo vendedor y probabilidad; no convertir los scores públicos de los cinco bloques en etiquetas individuales.
4. Para la última submission, elegir con regla escrita **antes** de observar su score: candidata con mayor ganancia conservadora confirmada y exposición pública controlada; si ninguna supera esa barra, preservar 48 como elección competitiva y sólo recomendar un diagnóstico acotado si su valor informativo para el cierre justifica el riesgo. La 48 ya puede seleccionarse como submission final sin volver a consumir cupo si la competencia retiene el mejor envío.
5. Guardar código, predicciones pareadas, CSV final, hash y checkpoint; informar a Ariel archivo exacto, diferencias frente a 48, evidencia a favor y límite. Ariel decide y sube a Kaggle.

La meta es mejorar el score base mediante train, y reservar las pruebas públicas para verificar una hipótesis ya fundada. El ranking actualizado no altera etiquetas ni justifica buscar flips de IDs por score.
