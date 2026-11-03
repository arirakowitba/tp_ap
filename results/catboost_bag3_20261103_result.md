# Bagging de 3 CatBoost sobre baseline fuerte (2026-11-03)

Código congelado antes de confirmar: `scripts/ablate_catboost_bag3_20261103.py`. Planes predeclarados para seeds seller-group 20261124, 20261125 y 20261126. 70.000 train originales, ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`. Características históricas word(1–2)+char_wb(3–5) TF-IDF OOF por seller, TE OOF y seller categórico. Base: el primer CatBoost; candidata: media de 3 seeds, pesos 1/3 fijados antes del confirm. Early stopping y umbral en tune; confirm sólo para medición.

| Seed | Confirm | Acc base → bag3 | Net | Fixes/breaks | Logloss base → bag3 | Net umbral común .5 | Net nuevo/visto | IC95 bootstrap seller |
|---|---:|---|---:|---:|---|---:|---|---|
| 20261124 | 12.543 | .953042 → .955274 | +28 | 63/35 | .126047 → .125050 | +16 | +16/+12 | [+8,+48] |
| 20261125 | 12.869 | .953376 → .954464 | +14 | 43/29 | .122632 → .120912 | +7 | +4/+10 | [−3,+31] |
| 20261126 | 12.350 | .951417 → .951903 | +6 | 37/31 | .128090 → .126470 | +3 | +9/−3 | [−13,+25] |

No se eligieron pesos en confirm. En seed 20261124, los umbrales tune de base/bag fueron .56/.495, pero bajo umbral común .5 el saldo siguió +16. Los tres logloss mejoraron. Las particiones usan el mismo dataset y pueden compartir IDs, así que los IC por seed no se combinan como independientes ni extrapolamos a score público. En test original 21.487/30.000 vendedores están vistos en train (71,6%), cercano a los confirm artificiales (~70–73%). Hay agregados no supervisados del extractor histórico calculados antes del split, comunes a ambos brazos.

Hashes confirm-ID: 20261124 `e6217419d4a22a36c82c5d95ddfd181c5ec5250ba3a691e539d41702464df6cb`, 20261125 `4675c5751be89247951321b59334600c95be3780efca319232bc4b852067a96e`, 20261126 `457259dfb6dea7ed802dd2d055136588d11210924bbf865e681d8f4323c3199f`. Hashes pares: `b964b058f08a0d4e61a3fcb30c1d033732c4620865d55f84cc5096c715925a0c`, `9ac5660e260bc105e9f2788c9db311c882fffb1d3364cbbed933b6db845b554e`, `b4458ef2a0df5350267e38737796ebd771af8c1f111aa763a09f55e58adcf477`.

Interpretación: ganancia de varianza modesta pero consistente contra un modelo idéntico de una sola semilla. Próximo: refit de 70k y mapa de discrepancias por ID frente a D1+D2; controlar procedencia, calibración de umbral y evitar traducir CV directamente en +aciertos públicos. No gastar Kaggle aún.
