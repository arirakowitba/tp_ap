# Ronda 80 — presencia binaria en predictor word

Se sustituyó únicamente el TF-IDF word del predictor OOF seller-group por presencia binaria (binary=True, sublinear_tf=False); char, LR C=2, FastText, HistGB de segunda etapa y umbral .475 quedan fijos. Hipótesis: la repetición de términos en el texto compuesto añade ruido para vendedores nuevos. Comparación pareada en a/b. El tune fue mixto y se detuvo antes del corte c.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base→binario |
|---|---:|---:|---:|---:|---:|
| a | −19 | 85/85 | 0 | [−26,+27] | .210628→.210548 |
| b | +24 | 76/74 | +2 | [−22,+25] | .205913→.207084 |

Dirección hacia new +8/+19, hacia used −8/−17. El log loss confirm b empeoró; ambos intervalos incluyen cero. No promover ni gastar confirm adicional. Los cortes ya se habían usado y se solapan.

Checkpoint `round80_binary_word_checkpoint_20261027.zip`, Library ID `libfile_f014074cd9208191a877b4a410312b0e`, SHA256 `29cae230a3149801b844abdcc8db3149de71f772f0c284f6243aa0bc766621f5`. Código, pares, bootstrap, métricas y manifiesto. Etiquetas solo train, sin Kaggle.