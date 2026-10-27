# Ronda 85 — cinco folds seller-group en predictores textuales OOF

Se cambió únicamente el número de folds OOF de word/char LogisticRegression de 3 a 5. El aprendizaje de vocabulario y etiquetas permanece dentro de fit; HistGB+FastText, tabular target encoding, semillas de split y umbral .475 se mantuvieron. Los cortes a/b/c ya son históricos y se solapan.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC bootstrap vendedor | Log loss confirm base→5 folds |
|---|---:|---:|---:|---:|---:|
| a | −4 | 101/82 | +19 | [−10,+49] | .210628→.209923 |
| b | +11 | 101/99 | +2 | [−30,+34] | .205913→.205836 |
| c, sin ajuste | +5 | 90/109 | −19 | [−48,+10] | .215678→.215605 |

Dirección hacia new +12,+17,−16; hacia used +7,−15,−3. En el subconjunto histórico de desacuerdos CatBoost (254/293/273 filas), candidato vs Cat +11/+10/+18, pero ese subconjunto fue seleccionado y no representa confirm completo. Los tres intervalos incluyen cero. No promover la variante ni seleccionar un umbral sobre confirm.

Checkpoint `round85_fivefold_text_checkpoint_20261027.zip`, Library `libfile_6126d79460f88191b1a3278666dec667`, SHA256 `d43bad3e9242858cc8b09c30059ef8f79b7b513b5681371dd2a543d865f205cc`. Incluye código, pares completos, métricas, bootstrap y manifiesto de dependencias. No se usaron labels de test ni se envió a Kaggle.