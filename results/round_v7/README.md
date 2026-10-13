# Ronda v7: mejoras temporales, corrección y poda

Fecha: 13-10-2026. Selección mediante folds agrupados seed 2026. Se mantuvo fijo el split; el holdout seed 731 no se reutilizó para elegir estos cambios.

## Resultados

| Experimento | Métrica | Decisión |
|---|---|---|
| TPE de referencia, 541 variables | 93,914% CV; log-loss 0,158692 | Baseline |
| Historial pasado del vendedor, como features, fold 0 | 93,877% y log-loss 0,158569; baseline fold 0: 94,011% y 0,155290 | No extender: peor accuracy y log-loss |
| Corrección: mezcla con seller prior temporal | Baseline 93,914%. Pesos 5/10/15/20%: 93,898/93,843/93,761/93,661% | Descartar; el log-loss también empeora |
| Poda del 20% de menor ganancia (108 columnas), 3 folds | 93,907% CV; log-loss 0,158497 | No promover: accuracy ligeramente menor, aunque mejora log-loss |

La poda varió por fold: accuracy 94,097%, 93,909% y 93,716%; el modelo completo obtuvo 94,011%, 93,979% y 93,753%. La pequeña mejora del primer fold no se sostuvo en el conjunto de folds.

## Conclusión

Ninguna de estas tres modificaciones mejora de forma consistente la accuracy del TPE. Se conserva el ensamble previo como mejor candidato y no se genera submission nueva. No se declara alcanzado el 95%.

## Reproducción

- `round_v7_experiments.py --mode correction`
- `round_v7_experiments.py --mode history`
- `round_v7_experiments.py --mode weak_features`
- `round_v7_cv.py`

El código usa los folds y matrices guardadas de `results/round_v6`, recalcula el historial solo con filas de entrenamiento anteriores en fecha y excluye publicaciones de la misma familia del historial de vendedor.

## Cribado por bloques (un fold agrupado)

| Bloque retirado | Variables retiradas | Accuracy vs. baseline | Log-loss vs. baseline |
|---|---:|---:|---:|
| Imágenes | 27 | 93,845% vs. 94,011% | 0,16015 vs. 0,15529 |
| Atributos | 250 | 93,979% vs. 94,011% | 0,15550 vs. 0,15529 |
| Texto | 25 | 93,400% vs. 94,011% | 0,16496 vs. 0,15529 |
| Vendedor | 16 | 93,604% vs. 94,011% | 0,16939 vs. 0,15529 |

Este fue un cribado breve en el fold 0 con iteraciones fijadas; ningún bloque retirado superó al baseline, por eso no consumí más folds con esas ablaciones. Se retuvieron todos los bloques en el pipeline.

Reproducción adicional: `round_v7_blocks.py`.
