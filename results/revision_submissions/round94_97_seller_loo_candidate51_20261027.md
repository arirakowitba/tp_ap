# Rondas 94–97 — nueva validación de memoria del vendedor y candidata 51 (27-10-2026)

La base sigue siendo HistGB+FastText con predictores textuales OOF y umbral .475. La señal adicional es la media suavizada de etiquetas de otros avisos del vendedor, calculada dejando fuera la propia fila en fit; en evaluación solo usa etiquetas de fit. Vendedores nuevos conservan exactamente la base. Se usaron las 70.000 etiquetas reales y los textos originales del conjunto de 100.000; jamás se asignaron etiquetas individuales a partir de scores públicos.

## Experimentos nuevos

R94 y R95 usan el mismo corte por filas (42.000 fit, 14.000 tune, 14.000 confirm) para comparar suavizado 5 y 20. En tune, 20 fue mejor en aciertos (+182 frente a +166 entre vistos), pero 5 dio mejor log loss. La selección congelada para el siguiente corte mantuvo 20 por accuracy y continuidad con rondas 90–93.

| Modelo | Tune vistos: neto vs base | Confirm vistos: fixes/breaks, neto | Log loss confirm vistos, base→modelo | Bootstrap 95% por vendedor |
|---|---:|---:|---:|---:|
| R94, suavizado 5 | +166 | 259/93, +166 (n=9.133) | .153752→.107322 | +124 a +210 |
| R95, suavizado 20 | +182 | 247/101, +146 (n=9.133) | .153752→.108053 | +104 a +190 |
| R96, suavizado 20, otro corte | +168 | 260/98, +162 (n=9.281) | .150816→.107426 | +119 a +204 |

En R96, los 4.719 nuevos quedan sin cambios. R94/R95 comparten confirm idéntico; R96 comparte 2.729/14.000 filas de confirm con R94 y 2.868/14.000 con R91. Por ello los resultados no son pruebas independientes. La comparación directa de 5 vs 20 sigue mixta: elegir 5 por R94 confirm sobreajustaría; queda 20 para la candidata.

## Candidata 51, sin score

Se entrenó suavizado 20 con las 70.000 etiquetas y se aplicó la compuerta solo a vendedores vistos. Se tomó el CSV **exacto** de la submission 48 (SHA-256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`) como política de referencia. La candidata cambia 373/30.000 predicciones, todas de vendedores vistos: 245 hacia used y 128 hacia new; 0 cambios en nuevos. CSV `submission_51_seller_loo_seen_on_48.csv`, SHA-256 `03a19e21a29d72800d7c0822afe5374966ad389dde6d0f9802889bdd667622fd`, Library `libfile_f778b44e5ecc819185b4af9865d235d9`. Checkpoint con código, predicciones pareadas y hashes: `libfile_b749acc455888191900a7898e9eb98e1`, SHA-256 `93e9ea00225975a99e23130f83b201407ca4404178ad8f6fc9dc6e954d004853`.

Diagnóstico, no labels: la base HistGB+FastText difiere de 48 en 994 avisos vistos; la compuerta en 373. La reducción de desacuerdos no predice accuracy pública. La comparación etiquetada frente a CatBoost para este subconjunto de vendedores vistos sigue pendiente, ya que los pares históricos disponibles se concentran en vendedores nuevos. La ventaja validada es frente a HistGB+FastText, no frente a la política 48; 48=.94633 es el mejor score comunicado, líder=.95404.

**Protocolo:** si se habilita cuota Kaggle y la cuenta es accesible, enviar primero la 51 como prueba del modelo de vendedores vistos, esperar su score, y decidir cualquier candidata siguiente según comparación agregada con 48. No hacer varias submissions correlacionadas antes de recibir el primer resultado. La subida no se realizó desde este entorno porque la sesión de Kaggle no está autenticada.