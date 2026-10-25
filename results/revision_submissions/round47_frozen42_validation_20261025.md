# Ronda 47: auditoría independiente de la regla congelada para la 42

Se fijó antes de abrir este corte la regla de la 42: cambiar `used→new` cuando HistGB predice used, FastText predice new y el margen `.475 - p_fasttext_used` es al menos .05. Se reentrenaron ambos modelos en un nuevo split seller-group, seed 46077. Fit: 41.229 filas; tune: 14.123; confirm: 14.648. Los tres conjuntos no comparten vendedores. Se usaron los textos originales de 100.000 filas y etiquetas solo de train.

| Segmento | Margen | Flips | Fixes | Breaks | Neto | IC 95% bootstrap vendedor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tune | ≥.05 | 114 | 62 | 52 | +10 | [−14,+33] |
| Confirm | ≥.05 | 81 | 47 | 34 | +13 | [−7,+32] |
| Tune | <.05 | 53 | 26 | 27 | −1 | [−16,+15] |
| Confirm | <.05 | 76 | 43 | 33 | +10 | [−7,+27] |

En todo confirm FastText sumó 45 aciertos frente a HistGB y redujo log loss .221879→.215678. Para la regla congelada, los netos son positivos en ambos segmentos, pero los intervalos por vendedor incluyen cero. Como el margen se escogió tras revisar otros cortes del mismo train, este corte nuevo aporta evidencia adicional, no garantía de transferencia. La comparación sigue siendo contra HistGB y no reproduce la política exacta de la submission 33. Por ello la 42 continúa como microprueba pública prioritaria; esperar su score exacto antes de recomendar la 43. No se envió ninguna submission.

`round39_disjoint_pairs_frozen42.npz` conserva fit/tune/confirm, etiquetas, vendedores y probabilidades pareadas. `round47_frozen42_audit.py` reproduce la segmentación y bootstrap.
