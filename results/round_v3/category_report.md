# Especialistas por categoría

Se probaron **25 configuraciones** sobre el mismo holdout por familia de 14.000 publicaciones (semilla 42). Incluyen un modelo global de referencia, especialistas para categorías con al menos 300/700 filas de entrenamiento, agrupamientos no supervisados de categorías según sus títulos (8/16 grupos, con y sin normalización L2), y mezclas con peso del experto 25%, 50%, 75% o 100%.

Los grupos se aprenden únicamente con títulos del entrenamiento. El especialista exige ambas clases y un mínimo de 30 ejemplos por clase. Categorías pequeñas o no vistas conservan la probabilidad del modelo global. Cada especialista usa un número fijo de árboles; no se realiza early stopping sobre el holdout externo.

|Comparación|Accuracy de selección|
|---|---:|
|Global con los mismos atributos y parámetros|93,9429%|
|Categoría exacta, mínimo 300, 25% experto|93,9643%|
|8 grupos sin L2, 50% experto|93,9643%|
|8 grupos con L2, 25% experto|93,8857%|

El mejor aumento son **3 aciertos de 14.000** frente al global del mismo experimento. Es una diferencia pequeña y todavía no confirmada. **No se logró 95% ni corresponde reemplazar la submission 94,428% con este resultado.**

La categoría exacta cubre 8,68% del holdout. Los agrupamientos con centroides sin normalizar terminaron casi todos en un grupo, por lo que su interpretación como categorías amplias es limitada. La normalización L2 produjo cuatro grupos importantes en la alternativa de 8 clusters, pero sus especialistas empeoraron el resultado. Con 16 clusters siguió existiendo un grupo dominante. Estas alternativas se conservan como evidencia negativa.

`category_metrics.csv` contiene todas las métricas; los JSON conservan parámetros, cobertura y especialistas; `category_selection_rows.csv` permite alinear las probabilidades NPY. `category_paired_diagnostics.csv` muestra errores corregidos e introducidos frente al global. Sus pruebas pareadas son exploratorias porque este holdout se usó para seleccionar modelos y umbrales.

La confirmación independiente corresponde a la semilla 731 y debe congelar parámetros, reglas y pesos antes de evaluarla.
