# Ronda 78 — ablación de C=8 solo en rama char

La ronda77 mejoró log loss pero no accuracy estable. Se dejó word en C=2 y se cambió únicamente char_wb 2–5 a C=8, con OOF seller-group, misma etapa HistGB+FastText y umbral .475. Se compararon pares en cortes a/b. Como tune no apoyó la variante y el log loss tune empeoró en ambos, se detuvo antes de c.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC vendedor | Log loss confirm base→candidato |
|---|---:|---:|---:|---:|---:|
| a | −5 | 84/86 | −2 | [−29,+23] | .210628→.210495 |
| b | +5 | 89/78 | +11 | [−16,+39] | .205913→.205482 |

Tune log loss: a .204014→.204074; b .203370→.203507. La pequeña ganancia confirm b procede de cambios hacia new +20 y hacia used −9; a +3/−5. Los intervalos incluyen cero y los cortes reutilizados se solapan. No promover ni ajustar basándose en confirm.

Checkpoint `round78_char_only_c8_checkpoint_20261027.zip`, Library ID `libfile_73562f61f9508191874797108390798e`, SHA256 `053d1aa09ab1c3f9a2cb30b9fd2b3494e11f45d047ab41da5e3e14926bb8c832`. Incluye código, pares, bootstrap, métricas y manifiesto. Solo etiquetas train. Ningún envío Kaggle.