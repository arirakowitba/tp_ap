# Auditoría de la rama Claude — 28-10-2026

HEAD observado: `07e2870c4049832c0873cdeeb05b8ac4f6115c3a` de `claude/posterior-y-modelos-20261027`. No se observan `round127_agente_a` ni `c` en ese HEAD; sí `b` y `d`.

Scores informados por Ariel: 48=.94633, 57=.94633, 58=.94633. Son mediciones agregadas redondeadas: no permiten inferir etiquetas de los 18 o 26 IDs cambiados. Se registran como bloques sin mejora observada.

`ens_ll8` comunica OOF random estratificado 5 folds .9518714 y logloss .1244056. Ocho semillas de CatBoost profundidad 5, corte por logloss; 59–62 son borradores disjuntos de 24/91/140/238 flips sobre 48, sin scores públicos.

## Hallazgo de validez

En `src/round126_modelos/train_gbm.py`, `te_oof` construye codificaciones de `seller_id`, categoría, ciudad e interacciones **una vez sobre los cinco folds globales** antes de entrenar cada CatBoost externo. Para el fold externo k, las filas de fit de otro fold j reciben TE calculado con las etiquetas de k. Por tanto, el modelo entrenado para validar k puede depender indirectamente de `y[k]` a través de las features de su fit. Las medias `stk_sellerother` calculadas de OOF de otros avisos también pueden contener modelos que vieron `y[k]`; `src/round127_agente_b/stack_perfold.py` reconoce explícitamente esta segunda fuga y propone predicciones por fold.

El OOF .9519 **no es una estimación libre de fuga**. No se cuantifica la magnitud del sesgo; puede ser pequeña o material. Además, `use_best_model` y early stopping por logloss seleccionan iteración con el mismo fold que se reporta, lo que aún optimiza sobre él. El aumento de .9519 respecto a baselines seller-group no demuestra mejora en vendedores nuevos: 71,6 % de test tiene vendedor visto, pero 28,4 % es nuevo. La CV aleatoria puede mezclar el mismo vendedor entre fit y validación, mientras los cortes seller-group son deliberadamente más exigentes.

Los archivos fuentes están descritos como 70k/30k JSONL recibidos de Hugo. Hay que verificar hashes/IDs y procedencia contra el original antes de comparar datasets. No se abrió ni usó el dataset público etiquetado MLA_100k mencionado en el informe del agente D; podría incluir respuestas de test.

## Condiciones para promover ens_ll8 o el stack B

1. Para cada fold externo, reconstruir TE y maestros de texto usando únicamente las etiquetas de fit de ese fold; las features para validación y test deben salir del mismo fit. Dentro del fit, generar predicciones OOF por folds internos sin tocar el fold externo. Para medias por vendedor de otros avisos, usar el modelo entrenado sin el fold externo para todas las filas que entrenan en el promedio.
2. Fijar hiperparámetros e iteraciones en tune separado; medir confirm por vendedor sin selección sobre ese confirm. Conservar pares con IDs y sellers para comparar ens_ll8, HistGB+FastText y CatBoost anterior en los mismos avisos, accuracy, logloss y bootstrap vendedor, separados seen/new.
3. Comparar los 59–62 solo como bloques de test sin etiquetas, y no atribuirles los `model_exp_net_total` como ganancias verificadas. El posterior público se limita al subconjunto público aproximado de 21000 y su feedback redondeado; no extrapolar al oculto ni convertir scores en labels individuales.

Conclusión provisional: no promover 59–62 como submissions basadas en una superioridad OOF demostrada hasta corregir este protocolo. Tampoco promover el stack B .9533, que usa la misma base OOF y folds aleatorios.

Checkpoint Library: `libfile_4d76ff581a1081919257081434ebe8f8`.
