# Ronda 135

# Ronda 135 — contexto no etiquetado del vendedor (28-10-2026)

HEAD Claude revisado: `612303c91d73645ccba5aa97504af43a13a64490`, sin nuevos commits desde la auditoría de ronda 134. Los informes A/B/C/D y la fuga de TE/OOF constan en los reportes anteriores.

Hipótesis: concentración de categorías y dispersión de precios del vendedor en los 100k avisos originales (sin etiquetas de test) pueden ayudar a una compuerta HistGB/FastText. Variables: log cantidad de avisos, log número de categorías, razón categorías/avisos, desviación del log precio, diferencia con mediana de precio del vendedor. Compresión `HistGradientBoostingClassifier` con logit HistGB y FastText. Fit y selección de mezcla separados por hash de vendedor dentro de tune; confirm permanece cerrado. Umbral .475 heredado.

| Corte | Peso congelado | Confirm n | Fixes | Breaks | Neto | Log loss base | Log loss propuesta |
|---|---:|---:|---:|---:|---:|---:|---:|
| a | base | 14533 | 0 | 0 | +0 | 0.210628 | 0.210628 |
| b | base | 14490 | 0 | 0 | +0 | 0.205913 | 0.205913 |
| c | 0.1 | 14648 | 7 | 11 | -4 | 0.215678 | 0.214986 |

En c, bootstrap por vendedor (3000 remuestreos) del neto: IC95% [-13,+4]. Los tres confirms históricos se solapan y no son réplicas independientes. Todos los holdouts son de vendedores nuevos; aquí no se evaluó vendedor visto. No hay pares CatBoost para todos los avisos de esos cortes. El log loss mejora en c pero accuracy cae; no se promueve el modelo ni se genera CSV.

Reproducción: `round135_seller_context.py`, `round135_report.json` y `round135_pairs_{a,b,c}.npz` en el checkpoint. Datos: `data.zip` original 70k train + 30k test sin etiquetas. Los scores públicos 48/57/58=.94633 solo se usan como historial agregado, nunca como labels.

Checkpoint SHA256: `ce126dd5a8aaf595ea071219e5a42c60a33cf160df382089b0398776d1811a8f`.
