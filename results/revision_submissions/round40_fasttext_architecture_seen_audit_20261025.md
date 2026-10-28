# Ronda 40 — arquitectura con FastText y auditoría de vendedores vistos

Fecha: 2026-10-25. Sin submission a Kaggle.

## Diseño

Se conservaron los 100.000 textos originales del Sheet, las 70.000 etiquetas de train, los vectores FastText sin supervisión de la ronda 39 y el umbral preexistente `p(used) >= 0.475`. No se usó ninguna etiqueta de test. La primera prueba añadió los 40 vectores a la segunda etapa CatBoost (700 árboles, profundidad 6, configuración fija de ronda 36), sobre los **mismos dos cortes disjuntos por vendedor** y las mismas predicciones de HistGB de ronda 39. La segunda prueba, nueva partición aleatoria de filas 40117 (42.000 fit, 28.000 holdout), entrenó HistGB con y sin vectores FastText; los docentes de texto y las codificaciones de categorías se entrenaron OOF agrupando vendedor. El holdout se desagregó por vendedores vistos en fit y no vistos.

## CatBoost + FastText en cortes disjuntos por vendedor

| Corte | Segmento | n | vs. HistGB sin FastText (fix/break/net) | vs. HistGB con FastText (fix/break/net) |
|---|---:|---:|---:|---:|
| 39002 | tune | 13.787 | 161/147/**+14** | 103/133/**−30** |
| 39002 | confirm | 14.533 | 166/142/**+24** | 139/127/**+12** |
| 39011 | tune | 14.335 | 156/137/**+19** | 144/143/**+1** |
| 39011 | confirm | 14.490 | 186/155/**+31** | 132/140/**−8** |

CatBoost + FastText mejora frente a HistGB sin FastText en cuatro segmentos, pero **no supera consistentemente HistGB + FastText**. Sus intervalos bootstrap por vendedor frente a HistGB sin FastText son, respectivamente, `[−25,+55]`, `[−15,+67]`, `[−17,+53]` y `[−11,+72]`; cada intervalo incluye cero. Frente a HistGB + FastText, la primera tune es `[−62,+1]`; los otros también incluyen cero. No se promueve la arquitectura ni se prepara una nueva submission a partir de ella.

## HistGB + FastText en partición de filas 40117

| Segmento del holdout | n | Hist correcto | FastText correcto | Fix/break/net | Log loss Hist → FastText | IC 95% neto por vendedor |
|---|---:|---:|---:|---:|---:|---:|
| Vendedores vistos | 18.606 | 17.537 | 17.559 | 133/111/**+22** | 0,151130 → 0,151457 | `[−10,+53]` |
| Vendedores nuevos | 9.394 | 8.363 | 8.404 | 122/81/**+41** | 0,265578 → 0,259810 | `[+14,+69]` |
| Todos | 28.000 | 25.900 | 25.963 | 255/192/**+63** | 0,189527 → 0,187809 | `[+21,+105]` |

La señal principal sigue concentrada en vendedores nuevos. En vistos, accuracy sube levemente pero no queda confirmada y log loss empeora ligeramente. La composición de este holdout es 66,5% vistos y 33,5% nuevos, frente a 71,6% y 28,4% en test; por eso el +63 agregado no representa una proyección al leaderboard. Esta partición comparte el universo de entrenamiento con otros ensayos y sus IC tampoco corrigen la búsqueda acumulada de modelos.

## Decisión y límites

Mantener HistGB + FastText como línea de investigación, con foco en vendedores nuevos y en la regla `used→new` validada en ronda 39. La candidata de 34 flips sobre la 33 sigue siendo **solo para revisión**. Ni esta prueba ni la 39 reproducen en train la política completa de la 33, por lo que no se debe trasladar el +41 o el +63 a expectativa pública. La segunda arquitectura evaluada no aporta una ganancia adicional estable. El siguiente experimento debería buscar una señal complementaria a FastText, evaluarla en una partición nueva por vendedor y después comprobar efectos en vistos y nuevos sin elegir filas por score público.

## Reproducción

Código: `round40_fasttext_cat_disjoint.py`, `round40_fasttext_seen_audit.py`, `round40_bootstrap.py`. Predicciones pareadas: `round40_fasttext_cat_pairs{,_confirm2}.npz`, `round40_fasttext_seen_pairs.npz`. Métricas: los tres JSON `round40_*report.json`. Dependencias de datos: `data.zip` SHA256 `f8e78574061cbd75197896fa1cd08a0c78af3cb5d73e7e2597df4ab43b82d574`, `original_docs.pkl`, `round39_fasttext_100k.npy` y dos pares disjuntos de ronda 39. El checkpoint incluye estos insumos y las fuentes compartidas. Python de entorno en `round29_environment.json`; CatBoost 1.2.10. Comandos: `PYTHONPATH=work20/deps python work20/round40_fasttext_cat_disjoint.py`, `python work20/round40_fasttext_seen_audit.py`, `python work20/round40_bootstrap.py`. Para reinstalar CatBoost usar `pip install catboost==1.2.10` en el entorno, si hace falta. El cálculo FastText original puede variar marginalmente con entrenamiento multihilo; se adjunta la matriz exacta de vectores.
