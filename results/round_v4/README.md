# Ronda 4: meta calibración de probabilidades

Esta ronda reutiliza solo las predicciones congeladas del ensamble anterior. Ajusta calibración logística de Platt e isotónica sobre el holdout agrupado seed 42 y transfiere cada calibrador y su umbral seleccionado a la confirmación agrupada seed 731. La confirmación no participa en el ajuste.

## Resultados

| Candidato | Umbral elegido en seed 42 | Accuracy seed 42 | Accuracy seed 731 | Log loss seed 42 | Log loss seed 731 |
|---|---:|---:|---:|---:|---:|
| Probabilidad original, umbral 0,500 | 0,500 | 94,2095% | 94,2143% | 0,15323 | 0,14651 |
| Probabilidad original, umbral ajustado | 0,499 | 94,2095% | 94,2071% | 0,15323 | 0,14651 |
| Platt, umbral ajustado | 0,482 | 94,2095% | 94,2071% | 0,15307 | 0,14670 |
| Isotónica, umbral ajustado | 0,429 | 94,2143% | 94,2071% | 0,15061 | 0,14858 |

La calibración isotónica mejora levemente la exactitud de selección y el log loss de selección, pero esa mejora no se transfiere a seed 731. Ninguna opción supera 95% ni mejora la referencia confirmada de forma material. No se promueve un nuevo modelo ni se genera una submission.

Los dos splits reutilizan filas del train del desafío y la familia del ensamble anterior: la confirmación sirve para elegir entre métodos ya definidos, pero no es un test externo independiente. Tampoco equivale a la puntuación pública de Kaggle.

## Reproducción

```powershell
python src/meta_calibration_round.py
```

El script guarda `meta_calibration_metrics.csv` y `meta_calibration_protocol.json`. Ajusta calibradores y umbrales solo con seed 42, y evalúa los valores congelados en seed 731.

## Por qué no se entrenó otro modelo base

Los archivos JSONL no están disponibles en las carpetas locales configuradas. El `data.zip` guardado en Library se localizó, pero su materialización no está disponible en este entorno. Sin las filas originales no puedo entrenar ni validar otro modelo complejo (por ejemplo, una nueva arquitectura neuronal o un nuevo ensemble de atributos), ni producir una submission nueva.
