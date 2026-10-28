# Ronda 126 — nueva familia de modelos (27–28 oct 2026)

Datos: `train_data.jsonlines` (70.000 filas, 53,7 % new) y `test_data.jsonlines` (30.000), recibidos de Hugo el 27-10. Mismo rango de fechas en train y test (2013-05-21 a 2015-10-15) y 71,6 % de las filas de test con vendedor visto en train. Por eso la validación principal es **CV estratificada aleatoria de 5 folds (semilla 2026)**, compartida por todos los modelos para poder mezclar sus OOF. Es más optimista que el holdout agrupado por vendedor/familia del repo; no comparar cifras entre protocolos.

Código en `src/round126_modelos/`. Configurar `AP2026Q2_DATA_DIR` con la carpeta de los JSONL. Orden: `feats.py` → `sparse_all.py`, `nbsvm.py`, `knn_title.py` (y opcionalmente `text_models.py`, `knn_price.py`) → `train_gbm.py` (o `run_queue.sh queues/queueN.txt LOGDIR`, una línea por experimento) → `average.py` / `blend.py` → `probe_eval.py` → `build_band_probes.py`.

Reproducir el mejor ensamble:

```
export AP2026Q2_DATA_DIR=/ruta/a/los/jsonl
python feats.py && python sparse_all.py 1.0 && python nbsvm.py && python knn_title.py
bash run_queue.sh queues/queue4.txt logs && bash run_queue.sh queues/queue5.txt logs && bash run_queue.sh queues/queue6.txt logs
python average.py ens_ll8 cat_d5_ll cat_d5_ll_b cat_d5_ll_c cat_d5_ll_lr03 cat_d5_ll_d cat_d5_ll_lr03b cat_d5_ll_e cat_d5_ll_lr03c
python build_band_probes.py ens_ll8 ../../submissions/20261028 59 "0.85,1.01" "0.7,0.85" "0.6,0.7" "0.5,0.6"
``` Las predicciones OOF/test quedan en `preds/pred_<modelo>.npz` (`oof` en orden de train, `te` en orden de ID 1..30000).


> **Aviso (28-10, hallazgo del agente B):** la columna `stk_sellerother_*` que agrega `train_gbm.py` filtra la etiqueta propia en el OOF (la predicción OOF de un compañero de vendedor viene de un modelo entrenado con y_i). Las cifras OOF de los modelos de esta ronda que la usan, incluido `ens_ll8` (0,9519), están **infladas**; en test no hay fuga. Sin esa columna, CatBoost d5 da 0,9520–0,9524 con un solo modelo (`b_cat_noso*`, ronda 127). Ver `results/round127_agente_b/README.md`.

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
| `txt_*` | 0,55–0,84 | vistas de texto separadas (título palabra 0,820, título char 0,817, garantía 0,627, atributos 0,551, combinada 0,841) |
| `cat_d5_txt7` / `b` | 0,9505 / 0,9502 | profundidad 5 con los 7 stacks de texto: no mejora a 2 stacks |
| `cat_d6_s24` / `cat_d4_s25` | 0,9498 / 0,9506 | otras semillas de profundidad 6 y 4 |
| `ens_d5x` | 0,9507 | promedio de 7 CatBoost profundidad 5; logloss 0,1272, la mejor |
| `ens_all` | 0,9508 | promedio de 11 CatBoost profundidad 4–6 |
| `lgb_l2` | 0,9495 | 2.º nivel con `ens_d5x` + media de su predicción en otros avisos del vendedor: no mejora |
| `cat_l2_d5` | 0,9520* | *no confiable: un fold eligió 1 iteración y logloss 0,212 |
| `ens_shallow` | 0,9505 | promedio de d5, d5 lr .03, d5 l2 10, d4, d6; mejor logloss (0,1279). Base para las pruebas del 28-10 |

Árboles más bajos generalizan mejor (profundidad 8 → 0,947; 4–6 → 0,950–0,951). La variación entre semillas (~0,1 punto) iguala las diferencias entre profundidades 4–6; promediar modelos no supera 0,9509. Siguiente: `knn_price.py`, precio relativo a los vecinos de título separados por etiqueta.

Errores de cat/v1 concentrados en vendedores con un solo aviso en train (11 % de error en v1, 28 % de las filas) y en `buying_mode` auction/classified.

## Contraste con las pruebas públicas

`probe_eval.py` cruza cada predicción de test con la posterior de la ronda 125. `cat_v1` difiere de 48 en 1.175 IDs: 610 ya probados (saldo público esperado −112 según la posterior) y 565 nunca probados. Esa cifra depende de la previa común q≈1/3 estimada sobre desacuerdos de modelos anteriores; para un modelo más fuerte puede ser pesimista. La forma directa de calibrar OOF→público es enviar una predicción pura del mejor modelo y comparar con 48.

## Sesgo de selección de iteración

Hasta `cat_l2_d5`, CatBoost usaba `eval_metric="Accuracy"` con `use_best_model` sobre el mismo fold que se reporta: la mejor iteración se elige con la métrica ruidosa sobre el fold de evaluación, lo que infla algo los OOF. Desde ahora `CAT_OD_METRIC=Logloss` (`cat_d5_ll*`) corta por logloss, métrica más suave; la diferencia entre ambos mide el sesgo.

Resultado: cortar por logloss entrena más iteraciones (1.800–4.400) y **mejora**; el sesgo de accuracy no inflaba las cifras.

| Modelo (corte por logloss) | OOF acc | logloss |
|---|---:|---:|
| `cat_d5_ll` / `_b` / `_c` | 0,9511 / 0,9512 / 0,9509 | 0,1267 / 0,1261 / 0,1261 |
| `cat_d5_ll_lr03` | 0,9513 | 0,1260 |
| `cat_d6_ll` / `cat_d4_ll` | 0,9499 / 0,9506 | 0,1290 / 0,1273 |
| **`ens_ll5`** (4 × profundidad 5) | **0,9519** | **0,1246** |
| `ens_ll_all` (+ prof. 4 y 6) | 0,9514 | 0,1249 |
| `cat_d5_ll_d` / `_e` / `_lr03b` / `_lr03c` | 0,9507 / 0,9506 / 0,9512 / 0,9511 | 0,1268 / 0,1273 / 0,1254 / 0,1256 |
| **`ens_ll8`** (8 × profundidad 5) | **0,9519** | **0,1244** |
| `cat_d5_ll_lr02` | 0,9503 | 0,1278 |
| `cat_d5_ll_txt7` (lr 0,03, 7 stacks de texto) | 0,9497 | 0,1279 |

`ens_ll8` es la base de las pruebas del 28-10 (`submissions/20261028/`): misma accuracy que `ens_ll5` con menor varianza de semilla en test.
