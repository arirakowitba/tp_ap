# Ronda 126 — nueva familia de modelos (27–28 oct 2026)

Datos: `train_data.jsonlines` (70.000 filas, 53,7 % new) y `test_data.jsonlines` (30.000), recibidos de Hugo el 27-10. Mismo rango de fechas en train y test (2013-05-21 a 2015-10-15) y 71,6 % de las filas de test con vendedor visto en train. Por eso la validación principal es **CV estratificada aleatoria de 5 folds (semilla 2026)**, compartida por todos los modelos para poder mezclar sus OOF. Es más optimista que el holdout agrupado por vendedor/familia del repo; no comparar cifras entre protocolos.

Código en `src/round126_modelos/`. Configurar `AP2026Q2_DATA_DIR` con la carpeta de los JSONL. Orden: `feats.py` → `sparse_all.py`, `nbsvm.py`, `knn_title.py` → `train_gbm.py` → `blend.py` → `probe_eval.py`. Las predicciones OOF/test quedan en `preds/pred_<modelo>.npz` (`oof` en orden de train, `te` en orden de ID 1..30000).

## Resultados OOF (accuracy, 5 folds)

| Modelo | OOF acc | Notas |
|---|---:|---|
| `nbsvm` | 0,8342 | NB-SVM n-gramas palabra/char de título + garantía |
| `sparse_all` | 0,9273 | Logística sobre TF-IDF título/garantía/atributos + one-hot de categóricas y numéricas binneadas |
| `lgb_v2` | 0,9390 | LightGBM, 63 hojas, + agregados de vendedor y stack `sparse_all` |
| `v1` | 0,9399 | LightGBM, 127 hojas, TF-IDF OOF de título y título+garantía+atributos, TE OOF vendedor/categoría/ciudad |
| `blend_a` | 0,9423 | Logística sobre logits de v1, lgb_v2, sparse_all, nbsvm |
| **`cat_v1`** | **0,9472** | CatBoost profundidad 8, `seller_id`/categoría/ciudad como categóricas, TE OOF, stacks `sparse_all`+`nbsvm` |
| `blend_b` | 0,9471 | cat_v1 domina la mezcla |
| `cat_v2` | 0,9474 | + vecinos por título (`knn_title.py`) |
| `cat_v4_s4041` | 0,9474 | cat_v2, otra semilla |
| `cat_v5_d6` | 0,9494 | profundidad 6, lr 0,05 |
| `cat_v9_d6s` | 0,9507 | profundidad 6, otra semilla |
| `lgb_v3` | 0,9468 | LightGBM con stack de cat_v1 |
| `cat_v8_d4` | 0,9504 | profundidad 4, lr 0,08 |
| `cat_v10_d5lr03` | 0,9507 | profundidad 5, lr 0,03 |
| **`cat_v7_d5`** | **0,9509** | **profundidad 5, lr 0,06; logloss 0,1285** |
| `cat_v11_d5l2` | 0,9505 | profundidad 5, l2 10 |
| `cat_v12_d3` | 0,9495 | profundidad 3 |
| `cat_v13_d5kp` | 0,9493 | + `knn_price.py`; peor logloss (0,1315): se descarta |
| `cat_d5_s21` / `cat_d5_s22` | 0,9509 / 0,9510 | profundidad 5, otras semillas: la configuración es estable |
| `cat_d5_nostk` | 0,9475 | sin stacks de texto: los stacks valen ~0,35 puntos → `text_models.py` agrega cinco vistas de texto separadas |
| `ens_shallow` | 0,9505 | promedio de d5, d5 lr .03, d5 l2 10, d4, d6; mejor logloss (0,1279). Base para las pruebas del 28-10 |

Árboles más bajos generalizan mejor (profundidad 8 → 0,947; 4–6 → 0,950–0,951). La variación entre semillas (~0,1 punto) iguala las diferencias entre profundidades 4–6; promediar modelos no supera 0,9509. Siguiente: `knn_price.py`, precio relativo a los vecinos de título separados por etiqueta.

Errores de cat/v1 concentrados en vendedores con un solo aviso en train (11 % de error en v1, 28 % de las filas) y en `buying_mode` auction/classified.

## Contraste con las pruebas públicas

`probe_eval.py` cruza cada predicción de test con la posterior de la ronda 125. `cat_v1` difiere de 48 en 1.175 IDs: 610 ya probados (saldo público esperado −112 según la posterior) y 565 nunca probados. Esa cifra depende de la previa común q≈1/3 estimada sobre desacuerdos de modelos anteriores; para un modelo más fuerte puede ser pesimista. La forma directa de calibrar OOF→público es enviar una predicción pura del mejor modelo y comparar con 48.
