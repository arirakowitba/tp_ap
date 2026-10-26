# Auditoría del proxy de la política 41

La submission 41 es la 33 exacta más seis decisiones conocidas de test: `new→used` en 1219, 8862, 22148 y `used→new` en 18412, 26663, 26668. Estas son decisiones enviadas, no etiquetas verdaderas. No hay en los artefactos una regla que produzca en train los 31 cambios manuales de la 33 respecto de 31 ni los seis cambios de 41. Por eso **no existe una evaluación etiquetada exacta de la política 41 en train**.

Se midió la fidelidad en las 30.000 decisiones de test de dos modelos ya preservados, sin usar score público como etiquetas:

| Proxy a umbral .5 | Discrepancias con 41 / 30.000 | Vendedores vistos | Nuevos |
| --- | ---: | ---: | ---: |
| `text_deep_cat_oof_fresh` | 552 | 239 / 21.487 | 313 / 8.513 |
| `text_deep_oof` | 576 | 241 / 21.487 | 335 / 8.513 |

Los seis cambios de 41 van en la misma dirección que ambos modelos; por eso reducen sus discrepancias frente a 33 en seis filas. Es una observación posterior a la prueba pública y no demuestra generalización a otros avisos. La concordancia global del primer proxy es 98,16%, pero entre vendedores nuevos baja a 96,32%; precisamente en ese segmento se buscan los cambios. Un análisis de gate contra ese proxy no mide automáticamente una mejora frente a 41. Debe informar por separado los 313 desacuerdos nuevos y la intersección de cualquier bloque propuesto con ellos.

Para comparación etiquetada, utilizar las predicciones seller-holdout OOF de cada arquitectura como baselines explícitos en train. Si se quiere aproximar la política 41 mediante destilación de sus decisiones en test, se debe medir fidelidad en test retenido, distinguir predicción de etiqueta y evaluar la estabilidad al pasar a predicciones OOF de train; no llamarla reproducción exacta de 41. No se entrenó tal destilador aquí porque no puede resolver los cambios manuales no especificados fuera de test y podría introducir una falsa sensación de exactitud.

Reproducción: `python3 work20/agent_proxy41_audit.py`. Datos originales train/test, CSV exacto de 33 y probabilidades `test.npz` preservadas en el checkpoint 44. No se utilizaron etiquetas de test ni se creó submission.
