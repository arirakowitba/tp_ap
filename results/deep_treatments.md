# Tratamientos robustos para AP_2026Q2

## Protocolo

Se conserva el holdout agrupado por `parent_item_id` (primeros tres folds de un `StratifiedGroupKFold` de 10 particiones, semilla 42). El objetivo es evitar que publicaciones hermanas aparezcan simultáneamente en entrenamiento y validación.

## Qué se agregó

- Variables profundas desde `shipping`, `variations`, `pictures`, `non_mercado_pago_payment_methods`, `attributes` y `seller_address`.
- Ratios de precio, stock disponible/inicial y stock vendido/inicial.
- Conteos, medias, desvíos y rangos de variaciones.
- Dimensiones y calidad de imágenes.
- Perfil de nulos sin usar `condition` como predictor.
- SVD TF-IDF del título y atributos, ajustado usando únicamente train.
- Experimento separado de flags IQR, calculados con cuantiles del desarrollo y sin reemplazar los valores originales.
- Corrección conservadora mediante prior de `category_id` solo en predicciones ambiguas.
- Target encoding out-of-fold de categoría y vendedor; se descartó porque empeoró el holdout.

## Resultados

| Modelo | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| CatBoost profundo, sin SVD | 0.93224 | 0.93297 | 0.93200 |
| CatBoost profundo + SVD, depth 7 | 0.93590 | 0.93643 | 0.93566 |
| CatBoost profundo + SVD, depth 8 | 0.93667 | 0.93720 | 0.93642 |
| CatBoost profundo, depth 6 | 0.93367 | 0.93427 | 0.93342 |
| CatBoost + SVD + umbral 0.49 | 0.93710 | 0.93777 | 0.93687 |
| CatBoost + SVD + prior de categoría | **0.93752** | **0.93819** | 0.93731 |
| Target encoding OOF de categoría/vendedor | 0.93552 | 0.93582 | 0.93525 |

Los flags IQR no se incorporan al ganador: el entrenamiento resultó demasiado costoso para el beneficio esperado y se detuvo antes de completar la ronda.

## Interpretación

El salto principal proviene de combinar la estructura de publicación con la representación semántica del texto. Aumentar depth de 7 a 8 ayuda a modelar interacciones como categoría × tipo de atributo × señales de stock/precio. El prior de categoría agrega una mejora marginal únicamente cuando el modelo está cerca del umbral y la categoría tiene historial suficiente y casi puro.

Para nulos se preserva una categoría explícita y se agregan conteos de ausencia. Para outliers se conservan los valores porque precios altos o stocks grandes pueden ser publicaciones legítimas. El target encoding numérico se descarta: CatBoost aprovecha mejor la información categórica original.

El holdout queda a 0.01128 de la referencia visible del leaderboard (0.94880). La próxima mejora de mayor valor debería venir de errores difíciles y representación semántica más rica, no de seguir agregando priors redundantes.
