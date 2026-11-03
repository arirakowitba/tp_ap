# LightGBM frente a la base histórica fuerte y bag3 — 2026-11-03

Se corrigió el problema de AG-0040: no se eligen ni peso ni umbral en confirm. Este experimento usa exclusivamente las 70.000 etiquetas originales del ZIP SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`, texto word+char OOF dentro de fit, TE desde fit, CatBoost bag3 pareado existente, y particiones seller testlike 20261124/25/26. LightGBM early stopping sólo en tune; umbral .5 y mezcla .5 LightGBM + .5 bag3 fijados antes de evaluar confirm. Código: `scripts/ablate_lightgbm_vs_bag3_20261103.py`. Pares y labels privados: Library `libfile_be6e6494a0088191a4000d829247e990`.

| Seed | N confirm | Bag3 acc / LL | LGB acc / LL | Mezcla fija acc / LL | LGB net vs bag3 | Mezcla net vs bag3 |
|---|---:|---|---|---|---:|---:|
| 20261124 | 12.543 | .955035 / .125050 | .945786 / .139943 | .953121 / .126075 | -116 | -24 |
| 20261125 | 12.869 | .954853 / .120912 | .950579 / .131426 | .955008 / .119688 | -55 | +2 |
| 20261126 | 12.350 | .951822 / .126470 | .947287 / .140840 | .952551 / .126402 | -56 | +9 |

LGB solo pierde en los tres cortes, especialmente con vendedores vistos: netos seen/new -101/-15, -61/+6 y -62/+6. La mezcla fija tiene netos seen/new -26/+2, -8/+10 y -1/+10. Sus intervalos bootstrap por vendedor son [-55,+7], [-27,+30], [-20,+39]. En conjunto hay 37.762 filas confirm pero sólo 30.017 item_id únicos; promedio del delta por ítem para la mezcla **-3,5**, IC bootstrap vendedor [-48,5,+39,3]. Para LGB solo: -176 ítems equivalentes, IC [-238,8,-114]. No sumar los tres cortes como observaciones independientes.

**Decisión:** LightGBM solo no es candidato; la mezcla .5 tampoco merece refit ni CSV con estos datos. AG-0040 no queda validado retrospectivamente: su combinación se seleccionó en confirm, comparó con una referencia más débil y usa otra arquitectura. El resultado aquí mide esta regla concreta, con comparador fuerte y corte pareado. El mejor score público demostrado sigue D1+D2 .94652. Se mantiene la búsqueda de una base que agregue decenas de correcciones robustas; no se gastó Kaggle.
