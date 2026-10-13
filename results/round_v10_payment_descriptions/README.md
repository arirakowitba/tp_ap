# Ronda v10: descripciones de medios de pago

## Pregunta y fuente

Una solución pública del desafío histórico de Mercado Libre conserva las descripciones de los medios de pago y comenta que aportan más detalle que el tipo general (`C`, `D`, `G`). La competencia actual evalúa accuracy. El modelo actual ya utiliza conteos y tipos de pago, pero no el texto exacto de esas descripciones; por eso se agregó una prueba aislada con el campo `description` concatenado y tratado como categoría.

Fuentes consultadas el 2026-10-13:

- Solución histórica y extracción de campos: https://github.com/ggnicolau/predict-used-new-marketplace
- Otra solución del dataset histórico de 100k: https://martinmjal.github.io/meli_challenge/solution.html
- Ficha pública de AP_2026Q2: https://www.kaggle.com/competitions/ap-2026-q-2
- Esquema oficial de productos y condiciones de Mercado Libre: https://developers.mercadolibre.com.bo/en_us/products-sync-listings/list-products

La ficha actual de Kaggle describe la tarea nuevo/usado y accuracy. La búsqueda no encontró código público verificable ni etiquetas del test actual de 30.000 filas. Los repositorios hallados son del problema histórico de `MLA_100k_checked_v3`, no una solución publicada a este test.

## Pruebas cortas

Se utilizó la misma configuración LightGBM y los mismos holdouts agrupados congelados que en la referencia. El holdout de selección tiene 56.000 filas de entrenamiento y 14.000 de validación. La confirmación seed 731 también tiene 56.000/14.000 y cero familias compartidas.

| Split / modelo | Accuracy @ .50 | Log-loss |
|---|---:|---:|
| Seed 42, normalizado base | 93,9214% | 0,155251 |
| Seed 42, descripciones solas | 93,9571% | 0,156229 |
| Seed 42, blend 10% @ .50 | 93,9429% | 0,155059 |
| Seed 731, ensemble previo | **94,2143%** | 0,146508 |
| Seed 731, descripciones solas | 94,1571% | 0,149642 |
| Seed 731, blend 10% con modelo normalizado | 94,2071% | 0,149189 |
| Seed 731, blend 10% con ensemble previo | 94,2143% | 0,146259 |

El blend con el modelo normalizado tuvo una señal de selección pequeña, pero no superó al mejor ensemble previo en la confirmación. El blend con el ensemble mejora log-loss, no accuracy. No se promueve ninguno; no se hizo búsqueda de pesos sobre el split de confirmación ni se cambió el CSV.

## Submission conservada

`submissions/submission_optimized_ensemble.csv` tiene 30.000 filas, IDs de Kaggle secuenciales únicos 1–30.000, sin nulos y solo clases `new`/`used`. El ID no coincide con el identificador MLA interno del JSONL porque el esquema de submission usa el índice de Kaggle. Sigue siendo el artefacto asociado al mejor score público informado por el usuario (94,428%); ese score no se reestima a partir de validación local.

Reproducción:

```powershell
python src/round_v10_payment_descriptions.py
python src/round_v10_payment_confirmation.py
```
