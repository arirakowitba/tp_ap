# Ronda 127 — Agente B: otros learners, texto y ensamble (28-10-2026)

Mismo protocolo que la ronda 126: 5 folds estratificados (semilla 2026) sobre los 70.000 de train, todo OOF.
Base: `ens_ll8` (promedio de 8 CatBoost prof. 5) **0,9519 / logloss 0,1244**. Diferencias de accuracy < 0,1 punto son ruido de semilla.
Código: `src/round127_agente_b/` (con `AP2026Q2_DATA_DIR=/home/claude/data`). Predicciones: `pred_b_<nombre>.npz` (`oof` orden train, `te` orden ID).

| Experimento | Idea | OOF acc | logloss | vs ens_ll8 | Veredicto |
|---|---|---:|---:|---|---|
| `b_lgb_t1` | LightGBM 31 hojas, l2 10, sin `seller_id`/`category_id`/`city` crudos (sólo TE OOF), misma tabla que CatBoost | 0,9438 | 0,1432 | −0,8 pts solo | más débil que CatBoost, pero útil en el apilado |
| stack `ens_ll8` + `b_lgb_t1` | logística sobre logits, CV anidada en los mismos folds | 0,9533 | 0,1217 | +0,14 / −0,0027 | **mejora**: peso negativo sobre LightGBM (−0,6) |
| `b_xgb_d5` | XGBoost hist prof. 5, lr 0,04, misma tabla, sin categóricas de alta cardinalidad | 0,9434 | 0,1425 | −0,85 pts solo | igual que LightGBM: débil solo |
| stack `ens_ll8` + `b_lgb_t1` + `b_xgb_d5` | logística C=0,1 sobre logits, CV anidada | 0,9535 | 0,1212 | +0,16 / −0,0032 | parece mejora, **pero ver fuga abajo** |
| `b_cat_pf` | cat_d5_ll (semilla 4141) con `stk_sellerother_*` **sin fuga** (ver abajo) | 0,9517 | 0,1266 | vs `cat_d5_ll` 0,9511/0,1267 misma semilla | igual o algo mejor y honesto; entra al promedio con peso positivo |
| `b_nn_full` | red desde cero (torch, 1 hilo): bolsa de tokens título/garantía/atributos/categóricas + CNN de caracteres del título | 0,8737 | 0,3040 | stack de texto | más fuerte que nbsvm (0,834) pero no aporta en el apilado final (peso 0) |

### Fuga en `stk_sellerother_*` (hallazgo)

`train_gbm.py` agrega `stk_sellerother_<s>` = promedio de la predicción OOF de `<s>` en los **otros avisos del mismo vendedor**. Para una fila i del fold k, la predicción OOF de un compañero j en otro fold viene de un modelo que **se entrenó con y_i**; los modelos lineales sobre one-hot de `seller_id` memorizan, así que la etiqueta propia se filtra al feature. En test no existe esa fuga. Ojo: la versión sin fuga es además *más informativa* (con `nbsvm`, exactitud del feature solo 0,876 → 0,904) porque los compañeros de train quedan predichos dentro de muestra, igual que en test. El tamaño de la fuga original no se puede medir directo; lo que se puede comparar es si las mejoras de apilado sobreviven con la versión honesta.
Arreglo (`stack_perfold.py` + `SELLEROTHER=perfold`): se re-entrenan `sparse_all`/`nbsvm` guardando la predicción de cada modelo de fold sobre todas las filas; para la fila i del fold k se usa el promedio de las predicciones del modelo k (que no vio y_i) sobre sus compañeros; para test, el promedio de los 5 modelos.

### Resultados con la tabla honesta

| Experimento | Idea | OOF acc | logloss | vs base | Veredicto |
|---|---|---:|---:|---|---|
| `b_cat_noso` | cat_d5_ll (semilla 4141) **sin** `stk_sellerother_*` | 0,9524 | 0,1250 | vs `cat_d5_ll` (misma semilla) 0,9511/0,1267: −0,0017 logloss, mejor en 4 de 5 folds | **mejor CatBoost individual** (iguala a `ens_ll8` con un solo modelo) |
| `b_lgb_pf` | LightGBM `b_lgb_t1` con sellerother sin fuga | 0,9428 | 0,1457 | — | débil solo |
| `b_xgb_pf` | XGBoost `b_xgb_d5` con sellerother sin fuga | 0,9427 | 0,1455 | — | débil solo, útil apilado |
| stack `b_cat_noso` + `b_cat_pf` + `b_xgb_pf` | logística C=0,1 sobre logits, CV anidada | 0,9539 | 0,1225 | +0,20 / −0,0019 | mejora sobre el promedio de los 2 CatBoost (0,9527/0,1245) en los 5 folds; gana 286 filas, pierde 200 |

Lectura: con la versión honesta el apilado con XGBoost **sigue** sumando (+0,12 pts sobre el promedio de CatBoost, mejor en los 5 folds), y la ganancia sigue concentrada en vendedores con 1–4 avisos en train. Agregar `te_seller` o su interacción con el conteo al apilador no suma.
| `b_cat_noso_lr03` | sin sellerother, lr 0,03, semilla 5252 | 0,9520 | 0,1253 | vs `cat_d5_ll_lr03` (misma semilla) 0,9513/0,1260 | confirma: sacar sellerother mejora con 2.ª semilla (−0,0007 logloss) |
| `b_lgb_noso` | LightGBM sin sellerother | 0,9423 | 0,1462 | — | no suma más que XGBoost en el apilado |
| `b_xgb_noso` | XGBoost prof. 5 sin sellerother | 0,9430 | 0,1457 | — | equivalente a `b_xgb_pf` |
| stack 2 cat noso + `b_cat_pf` + `b_xgb_pf` | logística C=0,1 | 0,9542 | 0,1214 | +0,23 / −0,0030 | mejor hasta acá |
