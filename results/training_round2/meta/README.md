# Validación de estabilidad del metaensamble

Se ejecutaron 3 repeticiones × 5 folds externos × 3 folds internos exclusivamente sobre las 2.098 filas del tune actual que también pertenecen al holdout histórico. Cada fold externo vuelve a elegir familia y parámetros dentro del entrenamiento externo. Se compararon histórico a umbral 0,5/0,52, calibración sigmoide, LR de seis logits y mezclas sparse con peso histórico >=0,5.

## Resultado tune nested

| Método | Accuracy externa media | Rango entre repeticiones |
|---|---:|---:|
| Histórico umbral0,5 |93,8513%| fijo |
| Histórico umbral0,52 |93,8036%| fijo |
| LR6, C elegido internamente |94,2326%|94,1373–94,3756%|
| Calibración sigmoide |93,8354%|93,7083–93,9466%|
| Mezcla sparse |93,8513%|93,7560–93,9943%|
| Procedimiento que elige también familia |94,0102%|93,8036–94,1373%|

La familia LR6 mejora al histórico en las tres repeticiones. El procedimiento completo de búsqueda tiene ganancia menor (+0,159 puntos contra histórico0,5) y una repetición inferior: hay incertidumbre por selección entre familias y muestra pequeña. La sigmoide no mejora, y las mezclas simples no ofrecen ventaja media.

## Selección y confirmación

Antes de cargar confirmación se guardó `selection_lock.json`; la selección tune-only vuelve a elegir LR6 C=1. En las 4.147 filas de confirmación comunes:94,2127%, logloss0,14876. Histórico0,5:93,9233%, logloss0,15281; histórico0,52:93,8992%. La LR suma12 netos frente0,5 (33corregidos/21rotos,p noajustada0,1337), y13frente0,52 (36/23,p0,1175).

Es exactamente el candidato anterior, no un nuevo ganador buscado en confirmación. `test.npz` contiene sus30.000 probabilidades, `selected_meta.joblib` el modelo y la configuración.

## Límites

La validación nested es del metaaprendizaje **condicional a predicciones base existentes**. Las arquitecturas, hiperparámetros y algunas representaciones de las bases ya se seleccionaron usando tune; estas capas no se reentrenaron dentro de cada fold. Tampoco se deshace la reutilización histórica de los holdouts. No es una validación end-to-end virgen ni acredita un score público. Las etiquetas de test y los saldos de flips no participaron.

El umbral0,52 se muestra por ser el histórico declarado. Su logloss en confirmación se calcula sobre las probabilidades originales; en nested el método0,52 expresa la misma decisión mediante traslado fijo del logit para usar umbral0,5, por lo que ese logloss se refiere a la probabilidad trasladada y no debe compararse como si fuera el original.

## Corrección posterior de procedencia
La auditoría posterior detectó que `test_probabilities.csv:refit49k_matched_without_public` corresponde a bases49k, sin equivalente full70k/core verificado. Por tanto, el test exportado aquí **no es una exportación full70k coherente** y no debe enviarse. Se preserva solo como diagnóstico. La evaluación tune/confirm no cambia por esta limitación de test. El estudio adicional `meta_production_consistent` usa la familia histórica `refit49k_without_public`, umbral0,505 y la probabilidad productiva full70k seleccionada; mantiene LR6 C1 por decisión previa, sin cambiar familia mirando confirm.
