# Ronda 52: mezcla CatBoost/FastText en vendedores nuevos

Se reutilizaron predicciones pareadas de tres cortes seller-group: CatBoost de segunda etapa y HistGB + FastText entrenados sobre el mismo fit. Se evaluó el holdout completo, separado en tune y confirm. Umbrales fijos CatBoost .5, FastText/mezcla .475; mezclas predeclaradas de 10%, 25% y 50% CatBoost. La tarea se puntúa por accuracy, así que se compara el saldo de aciertos y no solo log loss.

| Corte | Segmento | CatBoost solo − FastText (aciertos) | Mezcla 10% | Mezcla 25% |
| --- | --- | ---: | ---: | ---: |
| 1 | tune / confirm | −42 / +12 | +1 / +2 | −8 / +11 |
| 2 | tune / confirm | −17 / −18 | +9 / −3 | +26 / +1 |
| 3 | tune / confirm | −5 / −35 | +12 / +7 | +15 / −1 |

La mezcla 25% bajó el log loss en los tres confirm: .210628→.208463, .205913→.204135 y .215678→.213244. Sin embargo, los saldos de accuracy confirm fueron +11,+1,−1, y todos los intervalos bootstrap por vendedor incluyen cero. Si se eligiera peso por mayor neto en cada tune de la grilla 0/.1/.25/.5, se escogerían .1/.25/.25, que en confirm dan +2,+1,−1. No hay mejora de aciertos transferible suficiente para promover.

El segmento donde CatBoost coincide con FastText contra HistGB mostró ganancias locales, pero CatBoost perdió globalmente frente a FastText en dos confirm. Esta distinción impide extender ese acuerdo a una mezcla o submission masiva. Los cortes se solapan y la política exacta de la 41 no es reproducible en train. No se utilizaron etiquetas de test ni cuota de Kaggle.

El script `round52_cat_ft_blend_audit.py` reproduce métricas, fixes/breaks, log loss e intervalos desde los NPZ pareados guardados. Los modelos completos y datos originales están en el checkpoint de ronda 50, `libfile_9338e392cd5c8191bb8fd75365667362`.
