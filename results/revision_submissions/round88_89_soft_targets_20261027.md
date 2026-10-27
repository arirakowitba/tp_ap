# Rondas 88–89 — etiquetas suavizadas con predictores OOF

Hipótesis: regularizar el objetivo de segunda etapa mediante una mezcla predeclarada 90% etiqueta real y 10% promedio de probabilidades word/char OOF, calculadas solo en fit. Misma matriz de entrada HistGB+FastText, umbral .475. R88 usó HistGBRegressor con pérdida cuadrática; R89 representó la etiqueta fraccional con dos copias ponderadas por fila y HistGBClassifier con log loss. Los cortes históricos a/b se solapan; no se usaron test labels ni scores públicos por ID.

| Variante/corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base→candidato |
|---|---:|---:|---:|---:|---:|
| R88 regresor a | −74 | 113/188 | −75 | [−129,−27] | .210628→.229536 |
| R89 log loss a | −18 | 79/98 | −19 | [−51,+12] | .210628→.219160 |
| R89 log loss b | −11 | 88/131 | −43 | [−89,−2] | .205913→.216829 |

R88 fue detenida tras fallar el primer tune; R89 falló ambos tune. La versión con log loss mejora respecto al regresor, pero ambas quedan por debajo de la base. La pérdida se concentra en cambios hacia used: −61, −24 y −55 respectivamente. Se descarta esta mezcla de objetivos, sin ajustar el peso sobre confirm.

Checkpoint `round88_89_soft_targets_checkpoint_20261027.zip`, Library `libfile_64cf904c76708191b099efdba79e1ca0`, SHA256 `a9207075940e70801203747f4f97a18b281021a94e8224ac80a5580bc805ca94`. Código, pares, bootstrap y hashes. Sin submission ni cuota Kaggle.