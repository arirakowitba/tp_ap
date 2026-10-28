# Ronda 79 — C=8 solo para predictor word

Ablación de rondas 77/78: word TF-IDF LR C=8, char C=2, predicciones OOF seller-group; segunda etapa HistGB+FastText y umbral .475 idénticos. Se eligió C=8 por tune a/b (+8/+2 aciertos; log loss mejoró ambos) y se congeló para c. No se optimizó con scores públicos.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base→candidato |
|---|---:|---:|---:|---:|---:|
| a | +8 | 101/90 | +11 | [−17,+39] | .210628→.209234 |
| b | +2 | 96/108 | −12 | [−42,+19] | .205913→.206772 |
| c, congelado | +10 | 99/77 | +22 | [−4,+49] | .215678→.213793 |

Dirección hacia new: +6,+17,+8; hacia used: +5,−29,+14. En el subconjunto previamente seleccionado de desacuerdos CatBoost (254/293/273 filas), el candidato vs CatBoost +11/−2/+29; el subconjunto no representa todo confirm. Los cortes son seller-group (vendedores no vistos dentro de cada fit), se solapan históricamente. Al promediar predicciones repetidas por fila en los tres confirm hay 35.016 filas y 13.645 vendedores únicos, saldo +18 e IC por vendedor [−23.8,+59.7]; 7.423 filas aparecen dos veces y 616 tres veces. Esta señal es más interesante que la variante char, pero no es mejora confirmada; no preparar submission sin validación más independiente.

Checkpoint `round79_word_only_c8_checkpoint_20261027.zip`, Library ID `libfile_b6ca2dec9cd08191b8f05cc972f2cf99`, SHA256 `785c056932b89d7b166b448837aafc4bd60502ccd375ff29242d4704c2f432a4`. Incluye código, pares completos, métricas, bootstrap y hashes. Sin envío Kaggle ni cuota.