# Ablación CatBoost depth 8 frente a bag3 — 2026-11-03

En el corte seller-group predeclarado 20261124 entrené un CatBoost de profundidad 8, 1200 iteraciones máximas, learning rate .045, l2=8, early stopping y elección de iteración sobre tune. Usé exactamente las 70.000 etiquetas originales, la extracción word/char TF-IDF y OOF por GroupKFold, TE supervisado sólo desde fit y las variables seller/category de la arquitectura histórica. La partición contiene fit/tune/confirm 45.362/12.095/12.543. El ZIP de entrada verificado tiene SHA-256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`.

Se unieron por item_id las predicciones confirm privadas del bag3 y su primer modelo, verificando mismo orden y etiquetas. Umbral **fijo .5 para los tres brazos**, sin selección sobre confirm. Pares completos y etiquetas se guardan sólo en Library `libfile_bb2f67c3f2a881918d115a8850af8ee6`, SHA de CSV pareado `cd0bd1184610fec5006fc96fb16dff264cced3ad4ae39196b5feda2eddaaef69`. Código reproducible: `scripts/ablate_catboost_depth8_vs_bag3_20261103.py`.

| Modelo | Accuracy confirm | Logloss confirm |
|---|---:|---:|
| Histórico single depth 6 | .953759 | .126047 |
| Bag3 depth 6 | **.955035** | .125050 |
| Nuevo depth 8 | .954875 | **.124896** |

Depth 8 suma **14 aciertos** respecto del single histórico al umbral .5, pero pierde **2** frente a bag3 (48 fixes, 50 breaks); seen/new -1/-1; IC 95% bootstrap vendedor [-24,+21]. Su leve mejora de logloss frente a bag3 no se traduce en accuracy estable. Un blend .5 depth8 + .5 bag3 en este mismo confirm empata a bag3; esto es exploratorio sobre confirm reutilizado, no evidencia de promoción. La iteración elegida fue 1197/1200 y merece verificar sobre un corte nuevo si se retoma la línea.

**Decisión:** no producir CSV público de depth8. Bag3 mantiene prioridad de validación, pero su ventaja deduplicada tiene IC que cruza cero (ver `results/catboost_bag3_overlap_audit_20261103.md`). Buscar arquitecturas diversas antes de gastar cuota; el mejor score público demostrado sigue D1+D2 .94652. No se consultó Kaggle.
