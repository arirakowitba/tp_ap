# Experimentos incrementales AP_2026Q2

## Resultado ejecutivo

Se evaluaron 15 configuraciones sobre un holdout fijo de 21.000 filas (30%) separado por parent_item_id/fila, con estratificación de condition. El mejor resultado es **0,927619 de accuracy**: CatBoost enriquecido con SVD de texto, 75 atributos, señales de stock/ventas/fechas e interacciones vendedor-categoría. La meta 0,95 no se alcanzó en este ciclo; faltan 0,022381 puntos.

La validación es representativa del test en estructura (los parent_item_id no se solapan con test y los vendedores pueden reaparecer), pero el holdout se utilizó para selección repetida y por eso el score es optimista.

## Cinco familias propuestas para el siguiente ciclo

1. **ExtraTrees**: muchos árboles aleatorizados; puede capturar reglas no lineales de atributos, pero pierde frente a CatBoost cuando hay categóricas de alta cardinalidad.
2. **HistGradientBoosting**: boosting por histogramas, rápido con features numéricas; requiere una codificación cuidadosa de categorías.
3. **Calibrated soft-voting / stacking**: combina probabilidades de CatBoost, LightGBM, XGBoost y texto con pesos aprendidos en folds; mejora solo si los errores son complementarios.
4. **Modelo lineal char n-gram**: TF-IDF de caracteres + LogisticRegression/LinearSVC; robusto a faltas y variantes de títulos, pero limitado para interacciones tabulares.
5. **Red neuronal tabular (MLP/TabNet/FT-Transformer)**: aprende representaciones densas y atención; el dataset tiene suficientes filas, pero las pruebas MLP (0,887–0,895) muestran que árboles tienen mejor sesgo inductivo.

## Conclusión

La mayor oportunidad está en atributos de categorías específicas (automóviles, inmuebles y piezas), señales semánticas del título y generalización a vendedores no vistos. El siguiente experimento debe usar validación cruzada agrupada de 3 folds y un stacking out-of-fold; no conviene seguir probando hiperparámetros sobre el mismo holdout sin reservar un segundo control.
