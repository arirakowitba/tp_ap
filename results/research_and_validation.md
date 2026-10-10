# Investigación y estrategia experimental

Fecha: 2026-10-10. El objetivo del desafío es maximizar accuracy; meta del equipo: 0,94.

## Qué apareció en la búsqueda

- La competencia vigente es `AP_2026Q2`; su descripción pública la identifica como el TP3 de Análisis Predictivo y dice que el conjunto proviene de una entrevista técnica de ML. La página de competencia es [Kaggle AP_2026Q2](https://www.kaggle.com/competitions/ap-2026-q-2).
- Un repositorio público de una prueba técnica de Mercado Libre presenta el mismo target (`condition`: `new`/`used`) y describe un archivo fuente MLA de unas 100.000 publicaciones. El README dice que el JSONL no está distribuido por tamaño. Informa accuracy 0,8862 en un holdout de 10.000 filas. Revisamos su código de atributos: agrega `condition` a la lista de features cuando existe en el DataFrame, por lo que su resultado no es evidencia válida sin verificar que esa columna se haya quitado antes del fit. No copiamos código ni usamos ese score como benchmark.
- Kaggle también aloja un dataset antiguo “Mercado Libre Data Challenge” con un archivo `train_pt_70.csv`, pero la página pública no permite comprobar que sea el mismo split ni la misma tarea; no se usó.
- Encontramos un proveedor comercial que anuncia 950.000 publicaciones argentinas y etiquetas `Nuevo`, `Usado` y `Reacondicionado`; ofrece una muestra pequeña y cobra USD 849,99. La frescura y coincidencia temporal con este desafío no se pueden verificar. No se compró ni se incorporó.
- No apareció un dataset abierto y verificable que pueda aumentar de forma legítima el train sin riesgo de mezclar etiquetas del test.

## Inspección local y fugas

- Los JSONL locales tienen 70.000 filas de train y 30.000 de test. `condition` solo existe en train; no aparece un campo `ITEM_CONDITION` dentro de `attributes`.
- `parent_item_id` no se solapa entre train y test. `seller_id` sí: 7.053 de 15.019 vendedores de test aparecen en train. Por eso la validación de este ciclo separa grupos de `parent_item_id` y permite que vendedores reaparezcan, como en el test.
- Las features nuevas omiten la etiqueta y excluyen atributos cuyo ID/nombre identifica directamente condición. Se incluyeron campos disponibles en el JSONL oficial, como precio, cantidades, estado, logística, IDs de vendedor, atributos del producto y fecha.

## Protocolo y límite

Diez intentos en un holdout fijo de 21.000 filas separado por familias (49.000 filas de desarrollo; `StratifiedGroupKFold`, seed 42, folds 0–2 como holdout). El score final del mejor modelo es un resultado de selección, no una estimación intacta: se miró repetidamente el mismo holdout. La validación cruzada anterior corresponde a modelos/features anteriores y no confirma esta mejora. No se alcanzó 0,94 en los diez intentos; hay que confirmar el candidato en Kaggle con `sample_submission.csv` oficial o en nuevos folds agrupados antes de llamarlo final.

## Evidencia técnica para los notebooks

- En datos tabulares medianos, el benchmark de Grinsztajn et al. identifica ventajas inductivas de los árboles sobre redes profundas y propone desafíos para redes tabulares: ser robustas a variables poco informativas, conservar orientación y aprender funciones irregulares. [Paper, NeurIPS 2022](https://arxiv.org/abs/2207.08815).
- TabNet propone seleccionar variables secuencialmente con atención; es una alternativa interesante para aprender, pero no se ejecutó porque este entorno no tiene PyTorch/TensorFlow y no debía añadirse una dependencia de entrenamiento pesada para este ciclo. [Paper, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16826).
- La guía visual de MLP se apoya en el material de clase `Intro a Redes para datos estructurados.pdf`: suma ponderada, ReLU, softmax, entropía cruzada, mini-batches, Adam y early stopping.

## Scores locales

Los diez intentos y métricas por fila están en [`experiment_attempts.csv`](experiment_attempts.csv). Mejor: intento 9, CatBoost con 75 atributos, SVD del texto e interacciones vendedor×categoría, accuracy 0,9276 / balanced accuracy 0,9280 / macro F1 0,9273. El ensamble fijo del intento 10 quedó en 0,9274.
