# Ronda 41 — vecinos semánticos por aviso y categoría

Fecha: 2026-10-25. Sin submission a Kaggle.

## Hipótesis y diseño

Se usaron los vectores FastText sin supervisión de los 100.000 textos originales. Para cada aviso, FAISS buscó los 64 vecinos más similares entre avisos **etiquetados de train**, y se calcularon proporciones de `used`, similitud y proporción de vecinos de la misma categoría para vecindarios fijos 1/5/20/64. Estas 15 columnas se agregaron a HistGB + FastText. Para las filas de fit, las etiquetas de los vecinos se obtuvieron exclusivamente de los otros folds `StratifiedGroupKFold` por vendedor; para las filas de holdout solo se consultaron etiquetas de fit. No se empleó ningún score público para elegir etiquetas o umbrales. Se preservó el umbral previo `p(used) >= 0.475` y se comparó fila por fila contra las predicciones de HistGB + FastText de ronda 39 en dos cortes disjuntos por vendedor.

## Resultados

| Corte | Segmento | n | Corrige | Rompe | Neto | Log loss base → vecinos | IC bootstrap por vendedor del neto |
|---|---:|---:|---:|---:|---:|---:|---:|
| 39002 | tune | 13.787 | 98 | 115 | **−17** | 0,204014 → 0,202721 | `[−47,+12]` |
| 39002 | confirm | 14.533 | 103 | 108 | **−5** | 0,210628 → 0,209170 | `[−35,+27]` |
| 39011 | tune | 14.335 | 127 | 111 | **+16** | 0,203370 → 0,201794 | `[−15,+48]` |
| 39011 | confirm | 14.490 | 113 | 126 | **−13** | 0,205913 → 0,205515 | `[−48,+21]` |

La mejora de log loss ocurre en los cuatro segmentos, pero accuracy es inestable. La regla `used→new` que había sido prometedora con FastText en ronda 39 tampoco se traslada: sus netos son **−6, +5, +15, −2** en 96/111/133/124 flips respectivamente. Un promedio local de los 20 vecinos, usado como predictor aislado en el primer holdout, alcanza aproximadamente 80,2% de accuracy: las estadísticas vecinales no son una base alternativa fuerte.

## Decisión

No promover el modelo de vecinos ni generar una candidata de submission. La evidencia sugiere que los vecinos pueden mejorar estimación probabilística, pero no decisiones al umbral actual. Los intervalos incluyen cero y los cortes comparten el universo de train, por lo que tampoco constituyen cuatro réplicas independientes. Conservar las predicciones y características pareadas para probar futuras señales complementarias sin repetir esta arquitectura.

## Reproducción

`round41_semantic_neighbors.py` genera `round41_semantic_feature{,_confirm2}.npz`, `round41_semantic_pairs{,_confirm2}.npz` y `round41_semantic_report.json`; `round41_semantic_bootstrap.json` contiene el remuestreo por vendedor (2.000 réplicas, semilla 41041). Dependencias: `data.zip` SHA256 `f8e78574061cbd75197896fa1cd08a0c78af3cb5d73e7e2597df4ab43b82d574`, `original_docs.pkl`, `round39_fasttext_100k.npy`, pares disjuntos de ronda 39 y fuentes compartidas. FAISS CPU 1.12.0; para reinstalarlo: `pip install faiss-cpu==1.12.0`. Ejecución: `PYTHONPATH=work20/deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python work20/round41_semantic_neighbors.py`. La matriz de vectores exacta y todos los insumos están en el checkpoint. No se usaron etiquetas de test ni cuota de Kaggle.
