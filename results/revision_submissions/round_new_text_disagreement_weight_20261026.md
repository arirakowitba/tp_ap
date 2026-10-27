# Ponderación por discordancia de profesores de texto

Dos splits seller-group conservados de ronda 39. Mismos 145 insumos HistGB+FastText (variables profundas, 40 dimensiones FastText, codificaciones OOF y predictores word/char OOF). Se asignó peso 1.75 a ejemplos fit con diferencia word/char OOF >.18 y distancia de alguno al umbral .5 <.25; los restantes peso 1.0. Configuración fija antes de mirar resultados; HistGB 380 iteraciones, 31 hojas, umbral .475.

| Split | Tune neto vs HistGB+FastText | Confirm neto | Log loss confirm base→variante | IC 95% vendedor neto confirm |
|---|---:|---:|---|---|
| 39002 | −5 (82/87) | +22 (91/69) de 14533 | .210628→.209519 | [−4,+48] |
| 39011 | +4 (97/93) | +9 (105/96) de 14490 | .205913→.206296 | [−21,+37] |

En confirm used→new aportó +14 en ambos; new→used +8 y −5. Los intervalos incluyen cero, el segundo log loss empeora y las particiones solapan train con experimentos previos. No hay validación frente a CatBoost pareado ni política exacta 48, así que no crear submission. Código, probabilidades por fila, vendedores e índices en checkpoint Library libfile_5905d53295248191a730be058d7add8d. Requiere data.zip original, original_docs.pkl, embeddings round39 y fuentes compartidas del checkpoint round39. Ninguna etiqueta de test ni score público se usó para entrenar.
