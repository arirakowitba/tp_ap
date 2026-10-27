# Ronda 65 — clasificador supervisado de embeddings FastText

Se entrenó HistGB solo sobre los 40 embeddings FastText originales, usando etiquetas reales de fit en dos cortes seller-group de ronda 39; tune y confirm no entran al ajuste. Se comparó con HistGB+FastText preexistente, mismas filas y umbral 0,475. Se probaron mezclas logit de 5%, 10% y 20% como diagnóstico; no se eligió una submission.

| Corte | Modelo | Neto tune | Neto confirm | Log loss confirm vs base |
|---|---|---:|---:|---:|
| a | embedding solo | −1706 | −1889 | .459811 vs .210628 |
| b | embedding solo | −1936 | −2224 | .477786 vs .205913 |
| a | mezcla 5% | −6 | −6 | .210552 vs .210628 |
| b | mezcla 5% | −2 | −3 | .206365 vs .205913 |
| a | mezcla 10% | −7 | −1 | .211062 vs .210628 |
| b | mezcla 10% | −8 | −17 | .207439 vs .205913 |

La representación sola pierde masivamente información textual y tabular. Una mezcla pequeña tampoco mejora la accuracy en confirm. Descartar esta arquitectura. Los cortes se solapan con historial previo y no son confirmación independiente. No usar scores públicos como etiquetas, ni preparar submission.

Checkpoint reproducible `round65_embedding_classifier_checkpoint_20261027.zip`, Library ID `libfile_c5153638ac6c819186a520ca145c4cd8`, SHA256 `f3f05a1520552e7c664e09f2b3b38dfd6965a7400007d211a304b94d2c55d5c8`. Contiene código, pares y manifiesto; requiere datos y embeddings de ronda 39.
