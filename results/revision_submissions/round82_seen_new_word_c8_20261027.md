# Ronda 82 — word C=8 con segmentos vendedor visto/nuevo

Se reentrenaron HistGB+FastText original y word C=8/char C=2 de ronda79 en un corte por filas nuevo, semilla 20261028. Fit 42.000, tune 14.000, confirm 14.000; etapa textual OOF agrupada por vendedor. Un corte por filas incluye vendedores vistos y nuevos en holdout, y no equivale a la validación seller-group de ronda81. Umbral fijo .475.

Tune +6 aciertos, log loss .190421→.189327. Confirm −3 (79 fixes/82 breaks), log loss .189059→.187531, IC bootstrap por vendedor [−28,+22]. Segmentos confirm:

| Segmento | N | Fixes/breaks | Neto | IC vendedor | Log loss base→C8 |
|---|---:|---:|---:|---:|---:|
| Vendedor visto | 9.195 | 42/44 | −2 | [−20,+15] | .147741→.146959 |
| Vendedor nuevo | 4.805 | 37/38 | −1 | [−18,+16] | .268125→.265172 |

C=8 mejora probabilidad en ambos segmentos, pero no accuracy. Los cortes previos muestran resultados mixtos; no promover como nueva base de submissions. Solo se usaron etiquetas de train; no hubo puntuaciones públicas por ID. El segmento nuevo es mucho menor que en seller-group y sus vendedores casi siempre aportan una sola fila.

Checkpoint `round82_seen_new_word_c8_checkpoint_20261027.zip`, Library ID `libfile_ade7d7628ad88191992d7789a38a3db3`, SHA256 `855708f680c828a48add028679a8fea966c623448a1ab15f53a6dbd6a37d37bb`. Código, pares, segmentos, bootstrap y hashes; ningún envío Kaggle.