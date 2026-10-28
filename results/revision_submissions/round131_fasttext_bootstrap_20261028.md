# Ronda 131 — bootstrap por vendedor de FastText frente a HistGB

Auditoría sin entrenamiento adicional sobre pares etiquetados existentes de ronda39, todos de vendedores nuevos. Umbral congelado .475. Remuestreo de 2000 conjuntos de vendedores (semilla 131) dentro de cada confirm; no se usaron scores públicos para ajustar nada.

| Corte | Filas | Fixes | Breaks | Neto | IC 95% bootstrap vendedor | LL HistGB | LL FastText |
|---|---:|---:|---:|---:|---:|---:|---:|
| a | 14533 | 133 | 121 | +12 | [−21,+45] | .212552 | .210628 |
| b | 14490 | 166 | 127 | +39 | [+3,+74] | .210532 | .205913 |
| c | 14648 | 159 | 114 | +45 | [+12,+80] | .221879 | .215678 |

Los tres cortes usan fit/holdout seller-disjoint, pero sus confirms solapan (a/b 2868, a/c 3218, b/c 3185; triple 616). No se deben tratar como tres réplicas independientes. Esta auditoría apoya la familia FastText para vendedores nuevos, especialmente log loss; no demuestra que los flips de una submission concreta superen a la política 48. No hay par CatBoost en estos mismos cortes para comparación directa.

Script y reporte: Library `libfile_3927c31aae7c8191844ecb4f8fcd8d71`; SHA256 `6e0dbc12c52c2daea349224585045db595993832d04b6bacdf41b20e5ddc8968`. Los pares originales se conservan en checkpoint ronda39.
