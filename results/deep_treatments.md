# Tratamientos robustos para AP_2026Q2

## Protocolo

Se conserva el holdout agrupado por `parent_item_id` (primeros tres folds de un `StratifiedGroupKFold` de 10 particiones, semilla 42). El objetivo es evitar que publicaciones hermanas aparezcan simult�neamente en entrenamiento y validaci�n.

## Qu� se agreg�

- Variables profundas desde `shipping`, `variations`, `pictures`, `non_mercado_pago_payment_methods`, `attributes` y `seller_address`.
- Ratios de precio, stock disponible/inicial y stock vendido/inicial.
- Conteos, medias, desv�os y rangos de variaciones.
- Dimensiones y calidad de im�genes.
- Perfil de nulos sin usar `condition` como predictor.
- SVD TF-IDF del t�tulo y atributos, ajustado usando �nicamente train.
- Experimento separado de flags IQR, calculados con cuantiles del desarrollo y sin reemplazar los valores originales.
- Escaneo de umbral de decisi�n sobre `P(used)`; se selecciona el umbral �nicamente en el holdout documentado y se reporta por separado para no confundirlo con validaci�n pura.

## Resultados disponibles

| Modelo | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| CatBoost profundo, sin SVD | 0.93224 | 0.93297 | 0.93200 |
| CatBoost profundo + SVD, depth 7 | 0.93590 | 0.93643 | 0.93566 |
| CatBoost profundo + SVD, depth 8 | **0.93667** | **0.93720** | **0.93642** |
| CatBoost profundo, depth 6 | 0.93367 | 0.93427 | 0.93342 |

Los flags IQR no se incorporan al ganador: el entrenamiento result� demasiado costoso para el beneficio esperado y se detuvo antes de completar la ronda. No se debe presentar como resultado comparable.

## Interpretaci�n

El salto principal proviene de combinar la estructura de publicaci�n con la representaci�n sem�ntica del texto. Aumentar depth de 7 a 8 ayuda a modelar interacciones como categor�a x tipo de atributo x se�ales de stock/precio, pero aumenta costo y riesgo de sobreajuste. Para nulos se preserva una categor�a expl�cita y se agregan conteos de ausencia; para outliers se conservan los valores y se prueban flags, porque precios altos o stocks grandes pueden ser publicaciones leg�timas.

El mejor holdout queda a 0.01213 de la referencia visible del leaderboard (0.94880). Todav�a no se alcanza 0.95; el siguiente paso de mayor valor es terminar la calibraci�n de umbral y evaluar una combinaci�n de probabilidades de depth 7/depth 8, no entrenar muchos modelos sin diagn�stico.

