# Auditoría del stack B — 28-10-2026

Rama Claude `claude/posterior-y-modelos-20261027`, HEAD `07e2870c4049832c0873cdeeb05b8ac4f6115c3a` observado en esta revisión, sin cambios desde la revisión anterior. Hay resultados `round127_agente_b` y `d`; no aparecen `a` ni `c` en el árbol de ese HEAD.

El agente B reporta `ens_ll8` CatBoost 8 semillas, CV aleatoria 5 folds: accuracy .9519, logloss .1244; `b_lgb_t1` solo .9438/.1432; stack logístico sobre logits .9533/.1217, coeficiente LightGBM ≈−.6. `stack_eval.py` ajusta el nivel 2 en cuatro folds OOF y predice el quinto, pero **no vuelve a entrenar el nivel base dentro de cada fold del nivel 2**. Cada probabilidad OOF de las cuatro partes de entrenamiento fue generada por un modelo de base que pudo usar etiquetas del quinto fold. Por tanto, los coeficientes del nivel 2 pueden depender indirectamente de etiquetas del fold evaluado. La CV del apilador es cruzada solo para el nivel 2; no es una CV anidada de la tubería completa.

`stack_eval.py` evalúa C=.01,.1,1.0 sobre los mismos folds y, con `--save`, guarda `oof` y `te` del **último C=1.0**, sin un procedimiento de selección externo. Si .9533 corresponde al mejor C elegido al mirar esas métricas, hay selección adicional sobre el mismo OOF; si corresponde a C=1, la cifra sigue afectada por el nivel base. Los promedios y la selección de umbral del script también usan esos OOF de base. No se encontró en el README una evaluación seller-group tune/confirm o bootstrap vendedor del stack.

La tabla de LightGBM (`frame.py`) repite la fuga ya identificada en `train_gbm.py`: TE global de cinco folds reutilizada dentro del fit de cada fold externo y `stk_sellerother` de predicciones OOF de otros avisos. `cat_b.py` agrega una opción `SELLEROTHER=perfold`, pero su propia docstring y código mantienen TE global y stacks OOF globales; esa opción corrige solo parte de la dependencia. Incluso el cálculo `perfold` asigna a cada fila de fit la predicción de su propio fold, cuyo modelo puede haber visto el fold externo que se evalúa.

Una evaluación válida requiere folds externos seller-group. Para cada uno, recomputar toda la tabla, codificaciones, TF-IDF/estudiantes y predictores de nivel 1 sin ninguna etiqueta del holdout externo; generar el train del nivel 2 mediante cross-fitting interno exclusivamente en fit. Fijar C, peso, umbral e iteraciones en tune separado y medir confirm intacto. Conservar pares de probabilidades de `ens_ll8`, `b_lgb_t1`, stack, HistGB+FastText y CatBoost anterior para los mismos IDs, con segmentos seen/new, logloss y bootstrap vendedor. Hasta entonces, .9533 no justifica promover 59–62 ni una submission del stack.

57=.94633 y 58=.94633 igualan a 48=.94633 públicamente; solo se registra efecto agregado de los bloques. Los borradores 59–62 tienen 24/91/140/238 flips disjuntos, score pendiente. El posterior de la rama está condicionado por un público aproximado de 21000 ejemplos y redondeo; no es verdad de test individual ni base para labels.

No se consultó ningún dataset externo etiquetado que pudiera contener test.

Checkpoint Library: `libfile_d09841e6bbd081918033602bf0ef943c`.
