# Ronda 146: meta logística de tres modelos fuertes (29-10-2026)

Predicciones históricas de ronda 13 alineadas por `row_id/y/seen`: HistGB, LightGBM y CatBoost, mismo holdout de 14k. Entrada: logits de las tres probabilidades, indicador seen y sus interacciones. LogisticRegression C=1 fijo, entrenada con etiquetas reales sólo en la mitad tune de un `GroupShuffleSplit` por seller; evaluación en mitad seller-disjoint. Ningún score público como label. Esta meta no usa las etiquetas de fit del modelo base salvo a través de sus predicciones externas.

Primera partición seed146: confirm 6.842; CatBoost accuracy .944022/logloss .147885, meta .944899/.147175; fixes/breaks 29/23, neto +6. Seen +4, new +2. Bootstrap vendedor IC95 del neto [−8,+20]. La media simple de tres bases baja accuracy .943438 aunque mejora logloss .146794.

Estabilidad con la **misma arquitectura e hiperparámetros**, reentrenada en cuatro particiones por vendedor:

| Seed | Confirm filas | Fix/break | Neto | Logloss Cat→meta | Seen net | New net |
|---:|---:|---:|---:|---:|---:|---:|
| 146 | 6.842 | 29/23 | +6 | .147885→.147175 | +4 | +2 |
| 2026 | 7.016 | 23/26 | −3 | .149772→.149296 | +3 | −6 |
| 314159 | 6.912 | 30/22 | +8 | .151636→.150697 | +8 | 0 |
| 883 | 7.015 | 25/32 | −7 | .146557→.147464 | −3 | −4 |

Los cortes se solapan: no sumar netos ni IC como observaciones independientes. Accuracy mixta y un logloss peor; mejora insuficiente para promover. Además todos proceden del holdout histórico ya explorado, no confirm virgen. No preparar submission; 48 permanece respaldo.

Checkpoint con código, métricas y pares: Library `libfile_82c576fb53e48191a58adfb8094fed46`; SHA256 ZIP `2661610029668e235ffcd6433ec1f3c52d79f00778a2696ac3d293a2d15f51d4`.
