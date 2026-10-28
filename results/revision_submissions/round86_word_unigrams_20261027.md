# Ronda 86 — eliminar bigramas word

Se entrenó un predictor word TF-IDF unigramas (1,1) en lugar de (1,2), con OOF seller-group, manteniendo char, FastText, HistGB y umbral .475. Hipótesis: los bigramas raros elevan varianza. Falló en los dos tune y se detuvo antes de c.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base→unigramas |
|---|---:|---:|---:|---:|---:|
| a | −26 | 105/105 | 0 | [−30,+32] | .210628→.212119 |
| b | −3 | 86/122 | −36 | [−68,−4] | .205913→.211312 |

Dirección hacia new +2/−8; hacia used −2/−28. La pérdida confirma que quitar bigramas no ayuda en este sistema; no seleccionar un umbral sobre confirm. Cortes históricos solapados, solo etiquetas de train.

Checkpoint `round86_word_unigrams_checkpoint_20261027.zip`, Library ID `libfile_300111c6b0cc81918449eaae61f814da`, SHA256 `2d952418cc6dfb190296439ad311e32b6578a496030f06a412981f8ce4d29346`. Código, pares, métricas, bootstrap y hashes. Sin submission ni Kaggle.