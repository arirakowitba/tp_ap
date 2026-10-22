# Metaensamble con procedencia productiva coherente

Corrección posterior al estudio matched/core. La auditoría encontró que su vector histórico de test correspondía a modelos49k, no al equivalente full70k. Este estudio conserva **LR6 C=1** elegido antes, sin buscar otra familia o regularización usando confirmación.

- Train meta:2.098 filas tune comunes; histórico `refit49k_without_public` y cinco bases nuevas.
- Confirmación común:4.147 filas.
- Histórico evaluado con su umbral declarado0,505:93,9715%, logloss0,155061.
- Meta a0,5:94,0921%, logloss0,149559.
- Corrige30 y rompe25: **+5 netos**, p pareada no ajustada0,5901.
- Test usa histórico `production_probabilities.csv:p_used_full70k_selected`, combinado con full70k de las cinco ramas nuevas. `test.npz` es el candidato coherente con ese procedimiento.

La mejora de accuracy es pequeña y no concluyente. La menor pérdida logarítmica sugiere una combinación de probabilidades algo mejor, pero no permite afirmar mejora pública ni confianza global. La corrección se realizó después de evaluar la versión matched/core; este no es un holdout nuevo. No se eligieron parámetros con confirmación ni se usaron saldos de flips o etiquetas test.

La tabla nested de `../meta/` se refiere a la variante matched/core original, no a esta familia. No transferir automáticamente sus porcentajes a este estudio. Para evaluar estabilidad de la variante productiva, repetir todo el nested usando without_public o entrenar una nueva partición end-to-end.
