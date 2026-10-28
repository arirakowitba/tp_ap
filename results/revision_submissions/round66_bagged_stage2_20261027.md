# Ronda 66 — ensamble de segunda etapa HistGB

Hipótesis: promediar un HistGB completo y dos entrenados con submuestras aleatorias del 80% reduce varianza sin cambiar los insumos. Se conservaron las variables tabulares, FastText 40 y predictores de texto OOF de la referencia; las etiquetas de tune/confirm no entran al ajuste. Dos cortes seller-group de ronda 39, ya usados anteriormente y solapados entre sí.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | Log loss confirm, base → ensamble | IC 95% neto por vendedor |
|---|---:|---:|---:|---:|---:|
| a | −3 | 46/46 | 0 | .210628 → .209776 | [−18,+20] |
| b | +6 | 53/62 | −9 | .205913 → .205775 | [−31,+12] |

La hipótesis de mejora de probabilidad tiene apoyo en ambos confirm, pero no hay mejora de accuracy ni soporte para promover cambios de clase. Los intervalos incluyen pérdidas. La selección pública sigue siendo 48=.94633; los scores de submissions no se trataron como etiquetas.

Checkpoint `round66_bagged_stage2_checkpoint_20261027.zip`, Library ID `libfile_17474a81762c819181151893db647a05`, SHA256 `a8fd98211506022592815af23553bd514d5271b46be32a3e83dd281848a76d40`. Contiene código, predicciones pareadas, métricas, bootstrap y hashes. Datos y embeddings originales en ronda 39 y data.zip.
