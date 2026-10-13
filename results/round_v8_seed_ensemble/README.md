# Ronda v8: ensemble de semillas

Se probó la configuración TPE ya seleccionada con una segunda semilla (`42`) en los tres folds agrupados seed 2026. Cada fold se guarda al terminar en `results/round_v8_seed_ensemble/seed42_fold{k}.npy`; el script reutiliza esos archivos si se reanuda.

| Variante | Accuracy OOF | Log-loss OOF |
|---|---:|---:|
| Baseline guardado (semilla 2026) | 93,9143% | 0,158692 |
| Segunda semilla sola | 93,8946% | 0,158478 |
| Promedio 50/50 | 93,9125% | 0,158210 |

El promedio baja el log-loss en los tres folds. Su accuracy con umbral 0,5 es prácticamente igual al baseline. Con umbrales elegidos usando los otros dos folds, el promedio 50/50 obtiene 93,8804% OOF frente a 93,8643% para el baseline con el mismo protocolo: una mejora de 0,016 puntos porcentuales, con variación entre folds (dos mejoran y el tercero empeora).

**Decisión:** conservarlo como candidato de calibración, sin promoverlo a submission de accuracy. No se hizo refit sobre las 70.000 filas ni se envió una submission. Se mantiene el mejor resultado público informado: 94,428%.

## Reproducción

```powershell
python src/round_v8_seed_ensemble.py
```

La métrica de folds usa las predicciones OOF existentes en `results/round_v6` y el mismo protocolo agrupado por familia. El test de 30.000 filas no tiene etiquetas oficiales disponibles localmente, por lo que no se infiere su resultado a partir de los folds.
