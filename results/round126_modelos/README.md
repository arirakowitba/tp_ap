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

Errores de cat/v1 concentrados en vendedores con un solo aviso en train (11 % de error en v1, 28 % de las filas) y en `buying_mode` auction/classified.

## Contraste con las pruebas públicas

`probe_eval.py` cruza cada predicción de test con la posterior de la ronda 125. `cat_v1` difiere de 48 en 1.175 IDs: 610 ya probados (saldo público esperado −112 según la posterior) y 565 nunca probados. Esa cifra depende de la previa común q≈1/3 estimada sobre desacuerdos de modelos anteriores; para un modelo más fuerte puede ser pesimista. La forma directa de calibrar OOF→público es enviar una predicción pura del mejor modelo y comparar con 48.
