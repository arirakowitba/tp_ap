# Ronda 104 — confirmación temporal del modelo de vendedor (27-10-2026)

Para probar la señal de vendedor fuera de los cortes aleatorios anteriores, ordené las 70.000 filas etiquetadas por `date_created`: 42.000 avisos antiguos en fit, los siguientes 14.000 en tune y los 14.000 más recientes en confirm. La segunda etapa mantuvo HistGB+FastText, textos originales y profesores OOF; el modelo alternativo agregó la media de otras etiquetas del vendedor con suavizado 20 y dejó a vendedores nuevos en la base. La propia etiqueta se excluye en fit. El umbral .475 y la compuerta se fijaron antes de mirar este confirm.

| Segmento | Tune: fixes/breaks, neto | Confirm: fixes/breaks, neto | Log loss confirm base→compuerta |
|---|---:|---:|---:|
| Vendedores vistos | 212/104, +108 (n=8.338) | 214/101, +113 (n=7.888) | .152469→.111713 |
| Vendedores nuevos | 0/0, 0 (n=5.662) | 0/0, 0 (n=6.112) | .272156→.272156 |
| Total | +108/14.000 | +113/14.000 | .204721→.181758 |

Bootstrap por vendedor del neto visto: **+70 a +159** (95%, 2.939 vendedores en confirm). El confirm temporal comparte aproximadamente 2.7–2.9 mil de sus 14 mil filas con cada confirm aleatorio previo; no es una réplica independiente. Aun así, demuestra que la señal conserva ganancia cuando se entrena en avisos anteriores y se evalúa en posteriores. No compara directamente con la política pública exacta 48 ni con CatBoost etiquetado de vendedores vistos.

Checkpoint con código, pares, métricas y hashes: Library `libfile_1260cf488b98819181f02e30a7c1e7c6`, SHA-256 `95117659dad815c9a5ee3e3974aa778527d6062e406dbc59e7bdf57d39164602`. Se mantiene candidata 51 preparada, sin score ni envío desde este entorno.