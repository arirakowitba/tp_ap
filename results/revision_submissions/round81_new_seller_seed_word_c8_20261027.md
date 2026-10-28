# Ronda 81 — corte seller-group adicional para word C=8

Se congeló la arquitectura word C=8/char C=2 de ronda79 y se creó un corte adicional con GroupShuffleSplit(seed=20261027), fit 60%, resto tune/confirm disjuntos por vendedor. Se reentrenaron tanto HistGB+FastText original como la variante desde el mismo fit; pares por ID y umbral .475. La semilla es nueva, pero las etiquetas de train han participado en investigación previa, de modo que no es un confirm enteramente virgen.

| Parte | N | Fixes/breaks | Neto | Log loss base→C8 |
|---|---:|---:|---:|---:|
| Tune | 13.367 | 93/90 | +3 | .208008→.207697 |
| Confirm | 14.386 | 106/112 | −6 | .213063→.211881 |

Confirm contiene 5.590 vendedores; IC 95% bootstrap vendedor [−36,+25]. Cambios hacia new +2 y hacia used −8. Se solapa con los confirm históricos a/b/c en 2.796/2.955/3.316 filas (Jaccard .107/.114/.129). Con los cortes previos, respalda una mejora probable de log loss, pero no una mejora reproducible de accuracy; no promover una submission ni ajustar umbral con los confirm.

Checkpoint `round81_new_seed_word_c8_checkpoint_20261027.zip`, Library ID `libfile_bde3d370143881918640020f4bf59b61`, SHA256 `9e32f952d4f22a29d44a3ac45fd7f43ffef2fe3e95c5b73a78501b67c17ce98e`. Código, pares, métricas, bootstrap, solapamiento y hashes. Sin Kaggle.