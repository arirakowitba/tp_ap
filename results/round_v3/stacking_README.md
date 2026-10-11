# Stacking aprendido sobre probabilidades OOF

Esta prueba es distinta de `optimize_stack.py`: aquel experimento agregaba una
probabilidad de texto Logistic OOF como variable de CatBoost/LightGBM. Aquí una
**regresión logística de segundo nivel** aprende a combinar tres modelos completos:

1. LogisticRegression con TF-IDF de palabras/caracteres y one-hot de cinco categorías.
2. LinearSVC con las mismas representaciones; su margen se transforma con sigmoid
   (esta transformación no implica que la probabilidad esté calibrada).
3. LightGBM con variables estructuradas, SVD de texto y tasas agrupadas suavizadas.

## Protocolo

Se requiere el train íntegro de 70.000 filas. La selección usa 56.000 filas para
entrenar y 14.000 para evaluar, con la división por familias `seed42`. Cada una de
las tres bases se reentrena dentro de tres folds agrupados de esas 56.000 filas.
El vocabulario, ranking de atributos, SVD y categorías se ajustan con el training
de cada fold. Las tasas supervisadas de LightGBM se calculan con otros folds
agrupados **dentro** de ese training; no se reutilizan matrices globales con
encodings supervisados. Se verifica por aserción que ninguna familia cruza cada
frontera de entrenamiento/validación.

La regresión meta usa las probabilidades OOF de los tres modelos. Se comparan
`C ∈ {0.01, 0.1, 1, 10}` y entradas en escala de probabilidad o logit. La selección
incluye un umbral entre 0.46 y 0.54. Los modelos base tienen hiperparámetros fijos;
no se usa el holdout para early stopping. Después de elegir `C`, representación
y umbral se congelan y se repite el protocolo con `seed731`, sin ajustar nada
con sus labels.

Los scores de selección son exploratorios: compartir repetidamente un holdout
entre búsquedas puede causar optimismo. El resultado público 94.428% informado
por el usuario mide otro conjunto y no debe confundirse con accuracy local.

## Archivos

- `stacking_metrics.csv`: tres bases y ocho configuraciones meta en selección.
- `stacking_selected.json`: ganador, configuración congelada y confirmación.
- `stacking_*_predictions.csv`: predicción individual con `row_index` para alinear.
- `stacking_seed*_oof.csv`: probabilidad OOF, fold y fila original de cada ejemplo.
- `stacking_coefficients.json`: coeficientes sobre entradas estandarizadas.
- `stacking_seed*_base.npz`: cache reproducible de probabilidades, sin modelos.

El CSV OOF es evidencia de separación y una entrada válida para análisis del
meta-modelo. No se deben sumar predicciones por orden sin unir por `row_index`.
Esta prueba no genera ni promueve una nueva submission automáticamente.

```bash
python src/learned_stacking_round.py --data-dir ../data --output results/round_v3
```

El límite `--deadline` acepta un timestamp UNIX. Se comprueba entre ajustes;
el coordinador puede interrumpir el proceso si el límite llega durante un fit.

## Resultado de la primera variante (tres bases)

El ganador meta fue `stacking_probability_C10`, con threshold 0.46:
**93.8643%** en selección y **94.0143%** en confirmación congelada. La mejor base
LightGBM empató su accuracy de selección usando su propio umbral 0.535; por tanto
esta primera variante no demuestra una mejora individual de accuracy. Tampoco
supera el 94.2143% de la confirmación local histórica del ensemble recomendado.

Se prueban dos extensiones en archivos separados, manteniendo intacta esa evidencia:

- `learned_stacking4_round.py`: cuarta base CatBoost (800 árboles, profundidad 7),
  reentrenada en los mismos tres folds agrupados, con sus propias transformaciones.
- `learned_stacking5_round.py`: quinta base Logistic sobre embeddings de un modelo
  multilingüe preentrenado congelado. Ajusta escalador y clasificador dentro de
  cada fold, y mantiene el orden original de las 70.000 filas.

Las extensiones reutilizan probabilidades OOF válidas de las bases ya entrenadas,
no una matriz de variables con encodings supervisados calculados globalmente.
Cada una conserva métricas, selección y confirmación con prefijos `stacking4_` y
`stacking5_`. Los pesos del encoder de embeddings se identifican por hashes.
