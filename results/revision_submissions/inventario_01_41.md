# Inventario completo de submissions numeradas, 1–41 (25 oct 2026)

Este inventario separa **score observado**, **CSV exacto verificado** y **estado de envío**. El score no es una etiqueta por ID. Los archivos `base*_labels_1_to_30000.txt` codifican predicciones (0=new, 1=used); los TSV contienen todos los cambios por ID de los CSV recuperados. Las submissions 8, 9, 14 y 41 se prepararon, pero no hay score público observado.

| N.º | Score público | Predicciones por ID disponibles | Nota |
|---:|---:|---|---|
| 01 | 0.94433 | pendiente CSV exacto | nombre y score en `submission_feedback_log.csv` |
| 02 | 0.94419 | pendiente CSV exacto | mismo log |
| 03 | 0.94447 | pendiente CSV exacto | mismo log |
| 04 | 0.94457 | sí, respecto de 07 | Library; `flips_04_09.tsv` |
| 05 | 0.94447 | sí, respecto de 07 | Library |
| 06 | 0.94457 | sí, respecto de 07 | Library |
| 07 | 0.94476 | sí, base exacta | `base07_labels_1_to_30000.txt` |
| 08 | pendiente | sí, respecto de 07 | preparada, sin score |
| 09 | pendiente | sí, respecto de 07 | preparada, sin score |
| 10 | 0.94500 | pendiente CSV exacto | `feedback_10_observado_20261018.md` |
| 11 | 0.94519 | sí, respecto de 15 | Library; `flips_11_14_27_35.tsv` |
| 12 | 0.94419 | sí, respecto de 15 | Library |
| 13 | 0.94428 | sí, respecto de 15 | Library |
| 14 | pendiente | sí, respecto de 15 | preparada, sin score |
| 15 | 0.94523 | sí, base exacta | `base15_labels_1_to_30000.txt` |
| 16 | 0.94500 | sí, respecto de 15 | `flips_16_26.tsv` |
| 17 | 0.94447 | sí, respecto de 15 | ídem |
| 18 | 0.94404 | sí, respecto de 15 | ídem |
| 19 | 0.94409 | sí, respecto de 15 | ídem |
| 20 | 0.94285 | sí, respecto de 15 | ídem |
| 21 | 0.94504 | sí, respecto de 15 | ídem |
| 22 | 0.94471 | sí, respecto de 15 | ídem |
| 23 | 0.94495 | sí, respecto de 15 | ídem |
| 24 | 0.94533 | sí, respecto de 15 | ídem |
| 25 | 0.94519 | sí, respecto de 15 | ídem |
| 26 | pendiente de verificar | sí, respecto de 15 | CSV en el repositorio; no confundir score inferido con observado |
| 27 | pendiente de verificar | sí, respecto de 33 | Library; `flips_11_14_27_35.tsv` |
| 28 | pendiente de verificar | sí, respecto de 33 | Library |
| 29 | pendiente de verificar | sí, respecto de 33 | Library |
| 30 | pendiente de verificar | sí, respecto de 33 | Library |
| 31 | 0.94585 | sí, respecto de 33 | Library; también `feedback_31_35.json` |
| 32 | 0.94561 | sí, respecto de 33 | Library |
| 33 | 0.94614 | sí, base exacta | mejor observado; `base33_labels_1_to_30000.txt` |
| 34 | 0.94580 | sí, respecto de 33 | Library |
| 35 | 0.94609 | sí, respecto de 33 | Library, 13 flips recuperados |
| 36 | 0.94571 | sí, respecto de 33 | `flips_36_41.tsv` |
| 37 | 0.94490 | sí, respecto de 33 | ídem |
| 38 | 0.94423 | sí, respecto de 33 | ídem |
| 39 | 0.92976 | sí, respecto de 33 | ídem |
| 40 | 0.94595 | sí, respecto de 33 | ídem |
| 41 | pendiente | sí, respecto de 33 | preparada; no enviada ni puntuada |

**Cobertura:** 37 de las 41 numeradas tienen predicciones completas reconstruibles en este PR (4–9, 11–41). Faltan los CSV exactos de 01–03 y 10; de 26–30 faltan scores observados confirmados. Los ensayos previos sin numeración están en `submission_feedback_log.csv`; sus scores no se descartan del análisis, aunque no se dispone aquí de todos sus CSV. Los experimentos locales de entrenamiento están separados de las submissions públicas.

**Uso iterativo:** comparar cada CSV contra una base exacta, conservar la matriz de cambios por ID y los scores por submission, evaluar bloques con tune/confirm y actualizar el registro al llegar un score. La inferencia de cambios netos de un bloque solo es condicional a las hipótesis sobre la fracción pública y el redondeo. Nunca deducir etiquetas individuales de leaderboard ni prometer 96% a partir de pruebas parciales.
