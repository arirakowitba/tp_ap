# Ronda 77 — regularización de predictores textuales OOF

Se entrenaron dos intensidades de regresión logística word y char sobre los textos originales, dentro de fit, con OOF seller-group y la misma segunda etapa HistGB+FastText. C original 2; se ensayaron C=.5 y C=8. Umbral fijo .475. Se eligió C=8 por tune a/b y luego se ejecutó c sin ajuste adicional. Los tres cortes ya se habían usado históricamente y se solapan.

C=.5 perdió en tune −65/−17 y empeoró log loss: descartado antes de c. C=8 obtuvo:

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base → candidato |
|---|---:|---:|---:|---:|---:|
| a | +11 | 115/107 | +8 | [−22,+40] | .210628→.209258 |
| b | +18 | 105/115 | −10 | [−43,+24] | .205913→.205688 |
| c, congelado | +14 | 105/108 | −3 | [−32,+26] | .215678→.214304 |

El log loss de C=8 bajó en los tres confirm, pero las decisiones de accuracy no mejoraron establemente. El saldo hacia new fue +4,+16,−14; hacia used +4,−26,+11. Comparación con CatBoost en el subconjunto seleccionado previamente de desacuerdos (254/293/273 filas): candidato vs Cat +6/+3/+28; no representa el confirm completo. Ningún IC excluye cero. No promover a submission ni ajustar umbral sobre estos confirm reutilizados.

Checkpoint `round77_text_regularization_checkpoint_20261027.zip`, Library `libfile_76102a19f4d08191a21d751b49a00886`, SHA256 `19d0900e4efbcca602ef5110501ab4e3fcc7d1bfa54aab71e08b2b29872eab4f`. Incluye código, predicciones pareadas, métricas, bootstrap y hashes de insumos. Etiquetas únicamente de train; scores públicos no se convierten en labels por ID. No se gastó cuota Kaggle.