# Integración de fuentes y prueba de semántica de garantía (Ronda v9)

## Qué se incorporó

Se revisaron las ocho fuentes de la materia: introducción, consignas, reducción de dimensión, redes para datos estructurados, competencias, árboles, SVM y bases de datos/APIs. Los métodos principales que recomiendan ya están representados en el historial del proyecto: LightGBM/CatBoost y ensembles de árboles, SVM lineal como meta-feature de texto, embeddings y redes para variables estructuradas, SVD/normalización, stacking, validación agrupada y selección por accuracy. En las rondas previas, las redes, embeddings y modelos alternativos no superaron al ensamble de referencia.

La señal nueva con fundamento de dominio fue el desglose de garantía en **tipo** y **plazo**. Se agregaron al cribado cuatro variables derivadas del texto: tipo semántico, duración normalizada a días y log, unidad, y presencia de tipo explícito. El texto se normaliza sin usar la etiqueta; las predicciones siguen separando familias.

## Resultado

| Validación | Modelo | Accuracy | Log-loss |
|---|---|---:|---:|
| OOF agrupado seed 2026 | Baseline TPE | 93,9143% | 0,158692 |
| OOF agrupado seed 2026 | TPE + semántica de garantía | 93,8857% | 0,158476 |
| Confirmación seed 731 | Ensamble anterior (baseline) | **94,2143%** | 0,146508 |
| Confirmación seed 731 | Modelo con garantía | 94,1071% | 0,150264 |

El blend 50/50 del modelo de garantía con TPE mantiene menor log-loss OOF (0,158476) pero baja accuracy a 93,8857%. En la confirmación, un blend 10% garantía / 90% baseline empata accuracy y mejora log-loss a 0,146285. Otro blend de 10% con el modelo LightGBM normalizado sube un solo caso: 94,2214% contra 94,2143%. Esa diferencia no se considera mejora robusta después de probar varios pesos; no se promueve ni reemplaza la submission existente.

## Información pública adicional

- La competencia AP_2026Q2 es de condición nuevo/usado y puntúa accuracy. La confirmación seed 731 es un split agrupado del train, no el test oculto de Kaggle.
- Encontré repositorios públicos del ejercicio anterior `MLA_100k_checked_v3`. Sus señales de warranty, vendedor, categoría, atributos y cantidades ya están cubiertas por el pipeline actual. El README de `ProductClassification` reporta 87,8% accuracy y 95,0% AUC; no es una solución que mejore el score actual.
- La documentación oficial de Mercado Libre define `condition` como nuevo/usado/reacondicionado y especifica la garantía mediante tipo y tiempo. Las respuestas actuales de API no se usaron para leer las condiciones de las filas de test ni para generar etiquetas.
- No apareció una copia pública verificable con las etiquetas de los 30.000 test. Los pocos IDs/labels visibles en ejemplos públicos no se usaron para entrenar ni evaluar.

## Reproducción

```powershell
python src/round_v9_warranty_semantics.py
python src/round_v9_warranty_confirmation.py
python src/round_v9_warranty_confirmation.py --retrain
```

Los resultados están en `results/round_v9_warranty_semantics/summary.json` y `confirmation.json`. El código agrega la idea al flujo experimental; el pipeline/submission de referencia permanece sin cambios porque no hubo una mejora robusta de accuracy.


## Comprobación adicional del blend pequeño

En seed 42, el 5% del modelo con garantía añadido al ensamble seleccionado alcanza 94,1000% con umbral 0,50, frente a 94,0714% del mismo ensamble a ese umbral. Al transferir ese blend congelado al seed 731, obtiene 94,1571% a 0,50 y 94,1929% a 0,51; el candidato base logra 94,1714% y 94,1929%, respectivamente. La ventaja de selección no se transfiere, así que se descarta la promoción. El mejor baseline de seed 731 sigue siendo 94,2143% para el ensamble previo.

El valor 94,2214% observado para una mezcla distinta en seed 731 representa un solo caso adicional y no se replicó en el otro split; no se trata como mejora confirmada.
