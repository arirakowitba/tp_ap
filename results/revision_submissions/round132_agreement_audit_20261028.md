# Ronda 132 — acuerdo FastText/CatBoost en desacuerdos etiquetados

Auditoría nueva de los pares de ronda50, sin reentrenamiento. Son solo desacuerdos HistGB/FastText de vendedores nuevos; predicciones de CatBoost y etiquetas reales de train están alineadas por fila. Umbral .5, bootstrap por vendedor 2000 remuestreos, semilla 132.

| Corte | FastText used→new: neto/n | IC 95% | FastText new→used: neto/n | IC 95% | FastText+CatBoost contra HistGB: neto/n | IC 95% |
|---|---:|---:|---:|---:|---:|---:|
| primero | +19/87 | [0,+37] | −9/119 | [−35,+15] | +12/76 | [−5,+31] |
| segundo | +40/108 | [+18,+61] | +1/115 | [−22,+24] | +29/79 | [+12,+46] |
| tercero | +23/127 | [0,+46] | +22/96 | [+2,+42] | +22/74 | [+6,+38] |

Log loss en used→new HistGB→FastText: .7463→.6299, .7721→.6129, .7468→.6602. CatBoost en ese mismo segmento: .7072/.7431/.7271. Los cortes solapan poco en este subconjunto: 14/15/14 filas por pareja, cero en los tres. Son subconjuntos condicionados por desacuerdo; no representan la distribución completa. La política exacta de submission 48 no está disponible en train, así que esto apoya la dirección de la candidata 57 pero no valida sus 18 IDs ni su score. Tampoco se infieren etiquetas de test o scores públicos por ID.

Script y reporte: Library `libfile_6dd8183b69748191814ef078fc0d826d`, SHA256 `9ce9250b38a8f12f65a749063339780a8c6fe5bda588a370521977cf94f1c499`. Pares originales en checkpoint ronda50.
