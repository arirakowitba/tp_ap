# Ronda 76 — predictor OOF de caracteres del título

Se prolongó la tanda con tres entrenamientos seller-group a/b/c. Se añadió a HistGB+FastText una columna de un predictor char_wb 2–5 entrenado solo con títulos y OOF por vendedor. Se mantuvieron los insumos originales, el umbral .475 y las particiones. Etiquetas solo de fit; no se usaron resultados públicos como etiquetas por ID.

| Corte | Tune neto | Confirm fixes/breaks | Confirm neto | IC 95% vendedor | Log loss base → candidato |
|---|---:|---:|---:|---:|---:|
| a | −18 | 82/62 | +20 | [−6,+47] | .210628→.211306 |
| b | +5 | 87/90 | −3 | [−31,+26] | .205913→.205459 |
| c | +13 | 71/75 | −4 | [−28,+20] | .215678→.215422 |

Dirección used→new: +18,+8,−6 en confirm; new→used: +2,−11,+2. En el subconjunto previamente seleccionado de desacuerdos CatBoost (254/293/273 filas), título frente a CatBoost +9/+12/+23, pero no representa toda la población confirm. Los tres intervalos incluyen cero. Los cortes se solapan históricamente (ver auditoría ronda75), por lo que esta variante tampoco justifica una submission. Se descarta esta forma de predictor aislado del título.

Checkpoint `round76_title_char_checkpoint_20261027.zip`, Library ID `libfile_1e616d9efbb88191b7c722eebe940476`, SHA256 `19cfbe08e55fab401a24b6209e94c571eca8bc67e47fe4b884e8d7adbe764de9`. Incluye código, métricas, pares, bootstrap y manifiesto de dependencias. No hubo envío ni consumo de cuota Kaggle.