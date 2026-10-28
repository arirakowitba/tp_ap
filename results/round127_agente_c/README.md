# Ronda 127 — Agente C: arquitecturas distintas de los árboles (28-10-2026)

Objetivo: modelos que no sean GBM (redes tabulares, ExtraTrees, k-NN, propagación de etiquetas) para sumar diversidad a `ens_ll8` (0,9519 / logloss 0,1244).
Mismo protocolo: 5 folds estratificados (semilla 2026) sobre los 70.000 de train, todo OOF. Diferencias < 0,1 punto de accuracy = ruido de semilla.

Código: `src/round127_agente_c/` (con `AP2026Q2_DATA_DIR=/home/claude/data`, todo con `nice -n 15` y 1 hilo). Predicciones: `pred_c_<nombre>.npz` (`oof` en orden de train, `te` en orden de ID).
Tabla de entrada: la misma que `train_gbm.py` (cache de B `b_frame_sparse_all_nbsvm_knn_title.pkl`: 162 numéricas + 10 categóricas, con TE OOF, stacks `sparse_all`/`nbsvm` y `knn_title`).

Cómo se mide la mezcla (`c_common.blend_eval`, `c_stack.py`, `c_gstack.py`), siempre con CV anidada en los mismos 5 folds:
- **wavg**: promedio de logits con peso elegido en 4 folds y aplicado al 5.º.
- **stack**: logística sobre [logit base, logit modelo].
- **gstack**: igual, pero los logits también se multiplican por tres indicadores de segmento (vendedor sin avisos en los otros folds, vendedor con 1–10, `buying_mode` ≠ buy_it_now), más los indicadores sueltos. Son 4·m + 3 coeficientes, que se ajustan sobre 56.000 filas.



## Aviso importante: fuga en `stk_sellerother_*`

El agente B encontró que `stk_sellerother_*` (en `train_gbm.py` y en la tabla cacheada) filtra la etiqueta propia en el OOF. Mis primeros modelos (`c_mlp1`–`c_mlp8`, `c_et1`) usan esa versión, y por eso **sus mezclas con `ens_ll8` están infladas** (`ens_ll8` también la usa). Desde `c_mlp10` agregué `SELLEROTHER=perfold` (la versión sin fuga de B, `b_perfold_*.npy`) a `c_mlp.py` y `c_sk.py`. La base honesta para comparar es **`b_cat_pf` (0,9517 / 0,1266)**, un CatBoost de una sola semilla sin fuga. Las mezclas con `ens_ll8` se dejan sólo como referencia.

## Resultados

"Mezcla" = logística anidada sobre [logit base, logit modelo]. "+ns" son los modelos sin ninguna columna de vendedor, que quedan libres de la fuga.

| Modelo | Idea | OOF acc | logloss | Mezcla con `ens_ll8` (con fuga) | Mezcla con `b_cat_pf` (honesta) | Veredicto |
|---|---|---:|---:|---|---|---|
| `c_et1` | ExtraTrees 300 árboles, hoja mín. 2, max_features 0,3 | 0,9407 | 0,1551 | 0,9523 / 0,1237 | — | débil, aporta poco |
| `c_et3_pf` | igual, sin fuga | 0,9404 | 0,1560 | 0,9525 / 0,1235 | 0,9521 / 0,1262 | no aporta sobre `b_cat_pf` |
| `c_et2_nosel` | ExtraTrees sin columnas de vendedor | 0,9318 | 0,1773 | 0,9523 / 0,1236 | — | no |
| `c_knn50` | k-NN 50 (distancia) sobre numéricas cuantilizadas | 0,8851 | 0,2756 | 0,9518 / 0,1244 | — | inútil |
| `c_lp_t20` / `c_lp_t20s` | propagación de etiquetas transductiva en un grafo train+test (vecinos de título; `s` agrega aristas de vendedor) | 0,7920 / 0,8194 | 0,457 / 0,403 | sin ganancia | — | inútil: `knn_title` ya captura la señal |
| `c_mlp1` | MLP 512-256 con embeddings, cuantil→normal | 0,9358 | 0,1651 | 0,9523 / 0,1239 | — | sobreajusta (corta en la época 3–8) |
| `c_mlp2` | MLP 256, más regularizado (dropout 0,45, wd 1e-3, mín. 5 por categoría) | 0,9393 | 0,1554 | 0,9522 / 0,1233 | — | mejor |
| `c_mlp3w` | + parte lineal "wide" sobre TF-IDF del título y la garantía | 0,9317 | 0,1786 | 0,9526 / 0,1238 | — | peor solo, no aporta |
| `c_mlp4_nostk` | `c_mlp2` sin stacks de texto ni kNN (ns) | 0,9309 | 0,1739 | 0,9519 / 0,1242 | — | no |
| `c_mlp5_s3` | `c_mlp2` × 3 semillas | 0,9401 | 0,1528 | 0,9521 / 0,1233 | — | — |
| `c_mlp6_nosel` / `c_mlp9_ple_nosel_s3` | sin columnas de vendedor (ns) | 0,9323 / 0,9324 | 0,1729 / 0,1731 | 0,9520 / 0,1237 · 0,9521 / 0,1230 | — | no |
| `c_mlpoff1` | MLP sobre el residuo de `ens_ll8` (offset) | 0,9518 | 0,1245 | = base | — | no aprende nada (corta en la época 0) |
| `c_mlp7_ple` / `c_mlp8_ple_s3` | + codificación lineal por tramos (PLE, 12 tramos por cuantiles) | 0,9394 / 0,9403 | 0,1551 / 0,1515 | 0,9524 / 0,1226 · 0,9527 / 0,1223 | — | PLE mejora el logloss de la mezcla |
| `c_mlp10_ple_pf_s3` | `c_mlp8` sin fuga | 0,9398 | 0,1534 | 0,9528 / 0,1219 | 0,9525 / 0,1253 | **aporta** |
| **`c_mlp12_ple_pf_h512_s3`** | PLE 16, 512-256, wd 1e-2, dropout 0,5, emb 12, sin fuga, 3 semillas | 0,9397 | 0,1553 | **0,9531 / 0,1214** | **0,9526 / 0,1252** | **mejor miembro individual** |
| `c_mlp13_pseudo_s3` | `c_mlp12` + pseudo-etiquetas de test con p>0,97 (≈19k filas por fold, del modelo de fold) | 0,9398 | 0,1548 | 0,9534 / 0,1217 | 0,9526 / 0,1253 | = `c_mlp12`; cuesta el doble: no |
| `c_res1_s3` | ResNet-MLP (3 bloques residuales, 192) + PLE, sin fuga | 0,9392 | 0,1553 | 0,9533 / 0,1218 | 0,9524 / 0,1253 | = MLP |
| `c_nn_avg4` | promedio de logits de mlp10, mlp12, res1, mlp13 | 0,9401 | 0,1532 | 0,9532 / 0,1214 | 0,9525 / 0,1252 | igual que el mejor solo |

Apilados guardados (honestos, sobre `b_cat_pf`):

| Predicción | Composición | OOF acc | logloss | vs `b_cat_pf` (0,9517 / 0,1266) |
|---|---|---:|---:|---|
| `c_stk_pf_mlp12` | logística [b_cat_pf, c_mlp12] (pesos +1,19 / −0,29) | 0,9526 | 0,1252 | +0,09 pts; 218 aciertos nuevos contra 159 perdidos (signo p=0,003); 4 de 5 folds a favor; 203 cambios en test |
| `c_stk_pf_nn` | logística [b_cat_pf, c_nn_avg4] (+1,21 / −0,30) | 0,9525 | 0,1252 | +0,08 pts, p=0,004; 221 cambios en test |
| `c_gstk_pf_nn` | logística con compuertas por segmento | 0,9526 | **0,1240** | mejor logloss; 273 cambios en test |
| (ref.) `c_stk_ll8_mlp12` | [ens_ll8, c_mlp12] | 0,9531 | 0,1214 | contra `ens_ll8`: +0,12 pts, p=0,0006, **pero la base tiene fuga** |

## Conclusiones

1. La red tabular (MLP con embeddings + PLE) es el único miembro no-GBM que aporta de forma consistente. Sola llega a 0,940, 1,2 puntos por debajo de CatBoost. En la mezcla entra con **peso negativo** (−0,3 sobre `b_cat_pf`, −0,5 sobre `ens_ll8`): la logística usa la diferencia de logits CatBoost − red para "estirar" a CatBoost donde la red discrepa, igual que pasó con LightGBM en B. La ganancia sin fuga es **+0,08/0,09 puntos de accuracy** (justo en el borde del ruido de semilla, pero significativa en el test de signo pareado y en 4 de 5 folds) y **−0,0013 de logloss** (−0,0026 con compuertas por segmento).
2. Todas las variantes de red (más ancha, ResNet, pseudo-etiquetas, 3 semillas, promedio de 4) se saturan en la misma ganancia: la diversidad viene de "red vs árboles", no de la arquitectura de la red.
3. No sirven: k-NN, propagación de etiquetas transductiva, ExtraTrees (con la base honesta), parte wide de TF-IDF, el MLP sobre el residuo y las pseudo-etiquetas.
4. Las compuertas por segmento (vendedor sin historial, 1–10 avisos, subasta/clasificado) mejoran siempre el logloss del apilado (~−0,001 a −0,002) y la accuracy en ±0,03: útil si se apila.
5. Siguiente paso natural (no lo hice por CPU): un `ens` de 8 semillas **sin fuga** (`SELLEROTHER=perfold`) como base, y sobre ella `c_mlp12` con la logística anidada. Contra la base de 1 semilla la red aporta +0,09; contra un promedio de 8 semillas, probablemente algo menos.


## Actualización 12:55 — sobre la base sin `stk_sellerother` (`b_cat_noso`, de B)

B mostró que sacar `stk_sellerother_*` da el mejor CatBoost honesto (`b_cat_noso` 0,9524 / 0,1250; `b_cat_noso_lr03` 0,9520 / 0,1253). La red aporta **más** sobre esta base que sobre `b_cat_pf`:

| Apilado (logística anidada) | OOF acc | logloss | Ganancia contra su base |
|---|---:|---:|---|
| base: [b_cat_noso, b_cat_noso_lr03] | 0,9523 | 0,1243 | — |
| [b_cat_noso, c_nn_avg4] → `c_stk_noso_nn` | 0,9532 | 0,1239 | +0,09 vs b_cat_noso; 203 contra 142, p=0,001, 5/5 folds |
| **[b_cat_noso, b_cat_noso_lr03, c_nn_avg4] → `c_stk_noso2_nn`** | **0,9536** | **0,1223** | **+0,13 pts y −0,0020 logloss** vs la base de 2 CatBoost; 268 contra 178, p<0,0001, **5/5 folds a favor**; 340 cambios en test contra el promedio de los 2 CatBoost; pesos +0,63 / +0,72 / −0,40 |
| igual con compuertas (gstack) | 0,9538 | 0,1208 | mejor logloss de todo lo honesto |
| [b_cat_noso, b_cat_pf, c_nn_avg4] gstack → `c_gstk_noso_pf_nn` | 0,9536 | 0,1213 | |
| `c_mlp14_noso_s3` (la red **también** sin `stk_sellerother`) | 0,9396 solo / 0,1554 | con b_cat_noso: 0,9532 / 0,1238 | = `c_nn_avg4`: a la red no le cambia |

**Recomendación**: base = promedio/apilado de CatBoost sin `stk_sellerother` (ojalá varias semillas), más `c_nn_avg4` (o `c_mlp12_ple_pf_h512_s3`) como miembro con peso negativo en una logística anidada. Comando: `python c_stack.py b_cat_noso b_cat_noso_lr03 c_nn_avg4 --save <nombre>` (o `c_gstack.py` para las compuertas).

## Reproducir

```
cd src/round127_agente_c; export AP2026Q2_DATA_DIR=/home/claude/data OMP_NUM_THREADS=1
# requiere b_frame_sparse_all_nbsvm_knn_title.pkl (src/round127_agente_b/frame.py) y b_perfold_{sparse_all,nbsvm}.npy (stack_perfold.py de B)
SELLEROTHER=perfold nice -n 15 python c_mlp.py c_mlp12_ple_pf_h512_s3 --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.5 --mincnt 3 --emb 12 --hidden 512   # ~25 min
SELLEROTHER=perfold nice -n 15 python c_mlp.py c_mlp10_ple_pf_s3 --ple 12 --seeds 3 --epochs 20 --lr 1e-3 --wd 1e-3 --drop 0.45 --mincnt 5 --emb 8 --hidden 256
SELLEROTHER=perfold nice -n 15 python c_mlp.py c_res1_s3 --arch resnet --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.4 --mincnt 3 --emb 12 --hidden 192
SELLEROTHER=perfold nice -n 15 python c_mlp.py c_mlp13_pseudo_s3 --pseudo 0.97 --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.5 --mincnt 3 --emb 12 --hidden 512
SELLEROTHER=perfold nice -n 15 python c_sk.py c_et3_pf et 300 2 0.3
python c_stack.py b_cat_pf c_mlp12_ple_pf_h512_s3 --save c_stk_pf_mlp12
python c_gstack.py b_cat_pf c_nn_avg4 --save c_gstk_pf_nn     # c_nn_avg4 = promedio de logits de mlp10, mlp12, res1, mlp13
```
Los scripts se niegan a sobrescribir un `pred_c_*.npz` que ya existe. Las colas que se usaron están en `queue*.sh` y los logs en `logs/`. La red no es determinista bit a bit (torch en CPU); se esperan diferencias de ±0,0005 en OOF.
Predicciones copiadas en `preds/`: `c_mlp12_ple_pf_h512_s3`, `c_mlp10_ple_pf_s3`, `c_res1_s3`, `c_mlp13_pseudo_s3`, `c_nn_avg4`, `c_et3_pf`, `c_stk_pf_mlp12`, `c_stk_pf_nn`, `c_gstk_pf_nn`, `c_mlp14_noso_s3`, `c_stk_noso_nn`, `c_gstk_noso_pf_nn`, `c_stk_noso2_nn`.
`SELLEROTHER=none` (en `c_mlp.py`/`c_sk.py`) saca esas columnas: `SELLEROTHER=none nice -n 15 python c_mlp.py c_mlp14_noso_s3 --ple 16 --seeds 3 --epochs 20 --lr 7e-4 --wd 1e-2 --drop 0.5 --mincnt 3 --emb 12 --hidden 512`.
