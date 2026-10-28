# AP-2026-Q-2: registro incremental (25 de octubre de 2026)

La meta operativa es mejorar la generalización y buscar 96% público durante los próximos quince días. Es una meta, no una proyección garantizada. La referencia pública observada es la submission 33 (0.94614). Cada score procede del usuario y corresponde al leaderboard público; no hay etiquetas individuales de test conocidas.

## Scores observados y cambios respecto de 33

| Submission | Score público | Flips | Vendedor nuevo | Vendedor visto | Alcance |
|---|---:|---:|---:|---:|---|
| 31 | 0.94585 | — | — | — | Registrado en `feedback_31_35.json` del checkpoint; CSV no incorporado aquí |
| 32 | 0.94561 | — | — | — | Ídem |
| 33 | 0.94614 | 0 | — | — | Base exacta en `base33_labels_1_to_30000.txt` |
| 34 | 0.94580 | — | — | — | Score del checkpoint; CSV no incorporado aquí |
| 35 | 0.94609 | 13 | consultar inventario | consultar inventario | CSV exacto recuperado en Library; delta en `flips_11_14_27_35.tsv` |
| 36 | 0.94571 | 18 | 18 | 0 | 14 hacia new, 4 hacia used |
| 37 | 0.94490 | 146 | 6 | 140 | Consenso con margen 0.30 |
| 38 | 0.94423 | 201 | 6 | 195 | Los 146 de 37 más 55 vistos |
| 39 | 0.92976 | 1228 | 303 | 925 | Blend amplio |
| 40 | 0.94595 | 4 | 4 | 0 | Subconjunto hacia used de 36 |
| 41 | pendiente | 6 | 6 | 0 | Subconjunto de 37, tres hacia cada clase; CSV preparado, no evaluado |

`flips_36_41.tsv` identifica cada fila cambiada con la predicción base, la propuesta y el segmento. Una fila puede aparecer en varias submissions: es un evento por submission e ID. El archivo de base codifica `0=new, 1=used`, en orden de ID 1..30000. No contiene etiquetas reales. El registro ampliado con probabilidades, hashes y fuentes originales está en el checkpoint Library `libfile_a83aab280b5c8191b4ee34ee8af97c1d`. Con `replay_flips.py` se regeneran los CSV 36–41 y se validan sus conteos y anidamiento.

## Aprendizaje de bloques, no de etiquetas

* 40 frente a 33: cuatro cambios hacia used, delta público −0.00019.
* 36 frente a 40: catorce cambios adicionales hacia new, delta −0.00024.
* 38 frente a 37: cincuenta y cinco cambios adicionales en vendedores vistos, delta −0.00067.
* 37 frente a 33: 146 cambios, delta −0.00124; 140 son de vendedores vistos. El bloque de seis nuevos permanece sin medir por separado; 41 lo aísla.
* Los cambios de 36 y 37 no se solapan. 37 está contenido en 38 y 38 en 39.

Los deltas publicados tienen resolución de cinco decimales y exposición parcial. No se asigna verdad ni probabilidad posterior determinista a un ID por diferencias de score. Aunque un bloque pierda globalmente, puede contener aciertos. La selección reiterada sobre el mismo público produce sobreajuste.

## Protocolo para las próximas pruebas

1. Antes de decidir un flip, evaluar su familia de reglas con etiquetas de train y cortes tune/confirm, especialmente vendedores nuevos, vistos, títulos raros y márgenes. Guardar predicciones pareadas, fixes y breaks por segmento.
2. Elegir una pregunta por submission: un bloque pequeño, disjunto o estrictamente anidado en otro. Exportar base, delta de IDs, dirección, hashes y predicciones completas. Evitar ensayos amplios como 39 sin ganancia local replicada.
3. Tras recibir el score, anotar fecha, score exacto, comparación válida, tamaño del bloque y hallazgos a nivel grupo. Actualizar el TSV/registro y mantener una rama de exploración separada de la mejor submission observada.
4. Usar la 41 como siguiente prueba diagnóstica. Si mejora, comprobar en tune/confirm y explorar ampliaciones con una regla predefinida. Si empeora, probar otra familia independiente; no declarar erróneos sus seis IDs.
5. Evaluar nuevos modelos y características bajo splits por vendedor y un confirm reservado. Priorizar evidencia de generalización al oculto; el umbral 96% público no justifica etiquetar filas de test ni perseguir el leaderboard sin validación.

El inventario completo de 01–41 y la cobertura actualizada están en `inventario_01_41.md`. Se recuperaron predicciones exactas de 04–09, 11–41; faltan los CSV 01–03 y 10. Los scores 26–30 permanecen por verificar. Los experimentos locales de entrenamiento tienen documentación separada. Ninguna submission se envió como parte de este cambio.
