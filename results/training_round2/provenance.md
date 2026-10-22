# Auditoría de procedencia del ensamble histórico

Auditoría realizada sin entrenar modelos ni modificar scripts existentes. Los scores indicados son diagnósticos internos, no scores Kaggle.

## Conclusión

Las intersecciones de 2.098 filas de ajuste y 4.147 de confirmación usadas por `train_historical_stacker_next.py` están efectivamente excluidas del entrenamiento de las bases históricas y nuevas correspondientes. No encontré solapamiento directo de estas filas con los conjuntos de entrenamiento base pertinentes. Eso no convierte la confirmación en virgen: los holdouts históricos se han reutilizado y la selección de las bases nuevas utilizó el conjunto tune del cual se extrae el ajuste del meta-modelo.

Hay dos correcciones concretas: la referencia histórica se evaluó a 0,5 en vez de su umbral congelado, y la exportación test del ensamble utiliza probabilidades históricas de una base entrenada en 49.000 filas, no la producción con base refitada sobre 70.000. La mejora diagnóstica persiste al corregir los umbrales, pero es pequeña y no concluyente.

## Particiones verificadas

Archivo histórico: `recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/partitions.npz`.
Archivo nuevo: `tp_ap/results/new_models_20261021/partitions.npz`.

| Nuevo conjunto | Histórico inner 39.200 | Histórico tune 9.800 | Histórico hold 21.000 |
|---|---:|---:|---:|
| fit 49.000 | 27.385 | 6.860 | 14.755 |
| tune 7.000 | 3.912 | 990 | **2.098** |
| confirm 14.000 | 7.903 | 1.950 | **4.147** |

El histórico `selected_refit` concatena inner+tune=49.000; excluye hold21k. Los correctores históricos se ajustan sobre tune9.800, también disjunto de hold21k. Las filas comunes elegidas pertenecen a old hold; el meta-modelo nuevo se ajusta solo en common tune2.098 y puntúa common confirm4.147. Las nuevas bases fit49k excluyen tune7k, y sus refits dev56k excluyen confirm14k.

Limitaciones: las bases nuevas eligen hiperparámetros con tune7k completo, que contiene las 2.098 filas del ajuste meta. Su CV interna es maquinaria de selección, no una evaluación independiente del pipeline completo. El hold histórico ya sirvió para numerosos diagnósticos; los scores pareados no acreditan generalización nueva. Estas particiones no garantizan separación por vendedor. La auditoría reconstruye el código y los índices preservados; no demuestra que todo artefacto heredado esté libre de cada posible decisión histórica posterior al holdout.

## Qué significan las columnas históricas

Fuentes: `prior_code/warranty/compose_warranty_posterior.py`, `train_warranty_text_catboost.py`, `posterior_transfer_holdout.py` y `finalize_warranty_round.py`, bajo el checkpoint recuperado.

| Columna | Base para hold/test guardados | Corrector | Umbral congelado |
|---|---|---|---:|
| `refit49k_matched_without_public` | selected_refit entrenado con old inner+tune49k | core, ridge100, public_weight0, shrink1 | **0,52** |
| `refit49k_without_public` | misma base selected_refit49k | with_complementary, ridge10, public_weight0, shrink1 | **0,505** |

`matched_without_public` se construye copiando los demás hiperparámetros de la rama pública y poniendo peso público a cero. La función `fit_residual` omite tanto las restricciones agregadas como las tres etiquetas verificadas cuando weight=0. Por ello no ajusta directamente sus coeficientes con labels públicos, pero la definición matched mantiene una elección de estructura/regularización/umbral vinculada a la rama pública. No debe describirse como protocolo completamente independiente de la evidencia pública.

`without_public` es el ganador dentro de las alternativas weight0 elegidas por desarrollo. Sus settings coinciden exactamente con `selected` en el lock preservado. Es la referencia más apropiada para un componente sin corrector público directo.

## Corrección del umbral de la referencia

El script nuevo usa 0,5 para ambas columnas históricas. Sobre las mismas 4.147 filas:

| Referencia | Umbral | Aciertos | Accuracy | Ganancias/pérdidas del stacker | Neto stacker |
|---|---:|---:|---:|---:|---:|
| matched usada en informe anterior | 0,5 | 3.895 | 93,9233% | 33 / 21 | +12 |
| matched con umbral congelado | **0,52** | **3.894** | **93,8992%** | **36 / 23** | **+13** |
| without_public a 0,5 | 0,5 | 3.898 | 93,9957% | 37 / 28 | +9 |
| without_public con umbral congelado | **0,505** | **3.897** | **93,9715%** | **38 / 28** | **+10** |
| stacker nuevo | 0,5 | 3.907 | 94,2127% | — | — |

McNemar exacto sin ajuste por múltiples búsquedas: p=0,1175 frente matched al umbral correcto y p=0,2678 frente without_public al umbral correcto. La diferencia se mantiene, pero no es concluyente. Para logloss se conservan las probabilidades originales: mover el umbral no cambia esa métrica.

Introducir una probabilidad histórica cruda al meta-modelo logístico no obliga a convertirla primero según su umbral: el meta aprende su intercepto. El problema concreto está en la etiqueta de referencia y su comparación reportada, no en usar logits como input.

## Incoherencia del origen de test y mapping correcto

`composition/test_probabilities.csv` contiene `refit49k_*`. En `compose_warranty_posterior.py`, esas columnas se generan con `evaluation_predictions.npz['selected_refit_test']`, producido por `selected_refit.cbm` entrenado con **49.000** etiquetas. No es el modelo de producción70k. El ensamble nuevo mezcla ese componente49k con las cinco nuevas ramas70k en test; puede ser una mezcla válida, pero no debe describirse como todos los componentes refitados70k.

Existe archivo de producción:

`recovered/mercadolibre_checkpoint_20261017/mercadolibre_recovery_20261017/prior_results/warranty/results/production_probabilities.csv`

| Probabilidad para tune/confirm | Probabilidad coherente para test de producción |
|---|---|
| `composition/holdout_predictions.csv::refit49k_without_public` | **`production_probabilities.csv::p_used_full70k_selected`** |
| `composition/holdout_predictions.csv::refit49k_matched_without_public` | No hay equivalente full70k exportado en los archivos preservados |

La equivalencia del primer mapping se apoya en que `choices.selected` y `choices.without_public` son idénticos. `finalize_warranty_round.py` aplica el mismo `selected_corrector` congelado al modelo base70k y las nuevas garantías/teachers de producción. El archivo de probabilidades se escribe antes de aplicar los tres locks de labels al CSV submission; su columna selected no contiene esos locks duros. No usar `p_used_full70k_public_branch` como sustituto de matched sin público: su corrector sí tiene public_weight1.

La producción es híbrida incluso con la base70k: el manifiesto indica que nuevas garantías/texto se refitan con70k, mientras SVD y modelos complementarios viejos permanecen en sus fits de desarrollo. Describirla como **base70k con corrector y componentes auxiliares congelados**, no como entrenamiento70k de cada auxiliar.

Diferencia media absoluta entre `production.p_used_full70k_selected` y `composition.test.refit49k_without_public`: **0,01943562**. No son vectores intercambiables idénticos.

## Correcciones propuestas

1. Conservar los informes anteriores y añadir una corrección de umbral/procedencia, sin reescribir la historia de resultados.
2. Comparar cada referencia con su umbral ya congelado: matched0,52 y without_public0,505. No volver a elegir umbral en las4.147 filas.
3. Para un export producción coherente sin matched, volver a ajustar el meta usando `refit49k_without_public` en las2.098 filas de ajuste, confirmar una sola vez sobre4.147 con umbral histórico0,505, y exportar test con `p_used_full70k_selected`. No cambiar únicamente la columna de test del meta ya ajustado con matched: cambia la definición del input.
4. Si se mantiene matched como input, conservar su test49k y documentarlo; producir un matched70k exacto requeriría aplicar el corrector matched a las features de producción, no renombrar selected ni public_branch.
5. Antes de afirmar robustez, reservar evaluación adicional del pipeline completo con selección y calibración internas. Usar comparaciones pareadas y declarar reutilización histórica. Los próximos flips pueden aprovechar el candidato como nueva evidencia, pero no convierten este score interno en score público esperado.
