# Ronda 145: holdout 56k/14k de la tubería round126 + `b_cat_noso_b` (14-11-2026)

Holdout: `/home/claude/work/holdout/` (56.000 filas de train con etiqueta, 14.000 filas de train sin `condition` como test, ID = posición).
Para no pisar archivos del otro proceso que usaba `/home/claude/work/holdout/`, todo corrió en `/home/claude/work/r145_ho/`
(symlinks a los dos jsonl de ese holdout). Todo serial, 1 hilo. `holdout_labels.csv` se leyó sólo en `eval_holdout.py` y `summarize.py`, con modelos ya congelados.
Mismos 5 folds estratificados (semilla 2026) que la tubería original. SE del accuracy en 14k ≈ 0,19 pp.

## Resultados (OOF 5 folds sobre 56k vs holdout 14k con `te`)

| Modelo | Qué es | OOF acc | Holdout acc | Brecha (pp) |
|---|---|---:|---:|---:|
| `sparse_all` (original) | logística dispersa de round126 | 0,92404 | **0,83621** | **+8,78** |
| `sparse_fix` | ídem, con el binning corregido | 0,92816 | 0,92879 | −0,06 |
| `nbsvm` | sin cambios | 0,83154 | 0,83614 | −0,46 |
| `b_cat_noso_b` (original) | CatBoost exacto de round127 (SELLEROTHER=none, SEED=4242, CAT_LR=0.06) | 0,94746 | **0,93171** | **+1,58** |
| `b_fix_noso_b` | ídem con `stk_sparse_fix` en lugar de `stk_sparse_all` | 0,94841 | **0,94529** | +0,31 |
| `c_fix_inner` | `b_fix` con early stopping en un 10 % interno (no en el fold de validación) | 0,94689 | 0,94414 | +0,27 |
| `c_fix_nocat` | `b_fix` sin `seller_id`/`category_id`/`city` como categóricas nativas | 0,94061 | 0,94507 | −0,45 |
| `c_fix_tenest` | `b_fix` con `te_*`/`tecnt_*` anidados por fold externo | 0,94086 | 0,94321 | −0,24 |
| `c_fix_sknn` | `b_fix` + vecinos de título del mismo vendedor (OOF estándar) | 0,95075 | 0,94300 | +0,78 |
| `c_fix_sknn_nest` | ídem con sknn anidado (sólo cambian las filas de entrenamiento) | 0,94311 | 0,94414 | −0,10 |
| `c_fix_s2` | `b_fix` con semilla 5151 | 0,94770 | 0,94507 | +0,26 |
| `avg_fix2` | promedio de `b_fix_noso_b` y `c_fix_s2` | 0,94854 | 0,94536 | +0,32 |
| proxys LightGBM (`diag_lgb.py`) | `d_lgb_all` 0,93986/0,92807; `d_lgb_fix` 0,94061/0,94443; sin stk 0,93791/0,94407; sin etiquetas 0,92564/0,92564 | | | |

Validación interna "tipo test" (44,8k/11,2k dentro de las 56k, `make_inner_split.py`, sin tocar el holdout): `c_in_base` OOF 0,94529 / test interno 0,94134; `c_in_sknn` 0,94801 / 0,94045.

## Causa 1 (la grande, ~1,4 pp): binning de `sparse_all.py` depende del orden de fila

`src/round126_modelos/sparse_all.py:25`:
`pd.qcut(F[c].rank(method="first"), 20, ...)`. `method="first"` rompe empates por orden de aparición y el frame es `concat([train, test])`,
así que, para un valor empatado (binarios, ceros, precios redondos), las filas de train ocupan los rangos bajos y las de test los altos:
caen en bins distintos que nunca aparecen en train, o en el bin del otro valor. Ejemplo medido (`ship_free`): las filas de test con
`ship_free=0` quedan en los bins 15–19 y en el bin 19 están junto con las de train con `ship_free=1`. 77 de 107 columnas numéricas tienen filas de test
en bins sin ninguna fila de train (en los datos reales: 80 de 107, en promedio el 55,5 % de las filas de test por columna).
La CV no lo puede ver (la validación es parte de train). Efectos:
- `sparse_all` en el holdout: 0,836 vs 0,924 OOF; predice "used" en el 56,6 % (real 46,7 %).
- se propaga al CatBoost vía `stk_sparse_all`: `b_cat_noso_b` 0,9317 → `b_fix_noso_b` 0,9453 en el holdout (+1,36 pp; 331 filas ganadas y 141 perdidas, McNemar z = 8,75). En OOF la misma corrección casi no se nota (z = 1,85).
- **Datos reales:** el `pred_sparse_all.npz` de `/home/claude/data` tiene el mismo problema: el acuerdo con `b_cat_noso_b` baja de 0,956 en OOF a 0,909 en test (con `nbsvm` es 0,849 contra 0,850). Todo modelo enviado que use `stk_sparse_all` está afectado.
- Arreglo (`sparse_fix.py`): binning por valor (≤20 valores distintos: el valor mismo; si no, `qcut` sobre el valor, con los empates juntos). Con eso 0 columnas tienen filas de test en bins que no se vieron en train.
- El mismo patrón aparece en `src/round127_agente_a/sparse_raw.py:36` y `src/round127_agente_b/stack_perfold.py:18` (`b_cat_pf`), y en forma más leve en `src/round127_agente_a/build_feats_a.py:143` (cuartil de precio por categoría). Esos tres no los medí.

## Causa 2 (residual, ~0,3–0,8 pp de OOF optimista): features OOF no anidadas con los folds del GBM

Los `te_*`, `knn_*`, `stk_*` usan los mismos 5 folds que el CatBoost. Para el modelo del fold k, las features de **sus filas de entrenamiento** se calcularon
con etiquetas del fold k (validación). En test real eso nunca pasa (ninguna fila de train usa etiquetas de test). Lo probé así:
- `c_fix_sknn` vs `c_fix_sknn_nest`: **las filas de validación tienen exactamente las mismas features** (verificado en `sknn_nested.py`: idénticas salvo la escala de conteos); cambian sólo las filas de entrenamiento. OOF 0,95075 → 0,94311 (−0,76 pp) y holdout 0,94300 → 0,94414. Brecha +0,78 → −0,10. Con OOF estándar, sknn parecía mejorar (+0,23 pp OOF, mejor en los 5 folds, z = +5,0), pero en el holdout empeora (−0,23 pp, z = −2,55), y en el split interno tipo test también (−0,09).
- Con los `te_*` pasa lo mismo, pero sólo cuando CatBoost también usa `seller_id`/`category_id`/`city` como categóricas nativas. Con TE anidado (`c_fix_tenest`), el OOF cae 0,76 pp y la brecha pasa a −0,24, aunque el holdout también baja 0,21 pp (z = −2,33): las filas de entrenamiento quedan con TE de 3/5 de los datos. Sacando esas categóricas nativas (`c_fix_nocat`), el OOF cae 0,78 pp y el holdout no cambia (0,94507 vs 0,94529). O sea, lo que esas categóricas parecen sumar en CV es casi todo ilusorio. El mecanismo exacto de esa interacción no lo identifiqué: queda medido, no explicado.
- El early stopping sobre el fold de validación aporta poco: con ES interno, OOF −0,15 y holdout −0,11 (incluye entrenar con 90 %).
- Con features honestas, el holdout queda algo **por encima** del OOF (LightGBM −0,4 pp; `c_fix_nocat` −0,45 pp), porque en test las codificaciones usan el 100 % de train y no el 80 %.

## Mejor configuración

`b_fix_noso_b` (= `b_cat_noso_b` con `sparse_fix`): **holdout 0,94529**. `avg_fix2` (2 semillas): 0,94536, que es lo mismo (23 filas ganadas, 22 perdidas).
Ninguna de las mejoras probadas le gana por ≥0,15 pp en el holdout: sknn (estándar o anidado), sin categóricas nativas, TE anidado, ES interno.
Para elegir modelos en el futuro, la CV estándar de esta tubería **no sirve** para features por vendedor (sobreestima 0,3–0,8 pp y ordena mal, como pasó con sknn).
Hay que usar features anidadas (`c_fix_nocat` o `c_fix_tenest` tienen brecha ≤0) o el split interno tipo test.

## Archivos y reproducción

Código: `src/round145_claude_holdout/` — `sparse_fix.py`, `diag_lgb.py`, `cat_r145.py`, `seller_knn.py`, `sknn_nested.py`, `make_inner_split.py`,
`eval_holdout.py`, `avg_preds.py`, `summarize.py`, `run_base.sh`, `run_chain{1..8}.sh`.
Resultados: `results/round145_claude_holdout/metrics.json` (holdout), `metrics_inner.json` (split interno), `summary.json` (métricas, comparaciones pareadas, segundos por paso), `logs/`.
Predicciones: `/home/claude/work/r145_ho/pred_*.npz` (oof en orden de train, te en orden de ID), `/home/claude/work/r145_inner/pred_c_in_*.npz`.

```bash
mkdir -p /home/claude/work/r145_ho && cd /home/claude/work/r145_ho && ln -sf /home/claude/work/holdout/{train,test}_data.jsonlines .
export AP2026Q2_DATA_DIR=/home/claude/work/r145_ho OMP_NUM_THREADS=1 THREADS=1
cd /home/claude/tp_ap/src/round126_modelos && python3 feats.py && python3 sparse_all.py 1.0 && python3 nbsvm.py && python3 knn_title.py
cd ../round127_agente_b && SELLEROTHER=none SEED=4242 CAT_LR=0.06 python3 cat_b.py b_cat_noso_b                       # original
cd ../round145_claude_holdout && python3 sparse_fix.py 1.0
cd ../round127_agente_b && STACKS=sparse_fix,nbsvm SELLEROTHER=none SEED=4242 CAT_LR=0.06 python3 cat_b.py b_fix_noso_b   # corregido
cd ../round145_claude_holdout
NOCAT=1 python3 cat_r145.py c_fix_nocat; TE_NESTED=1 python3 cat_r145.py c_fix_tenest; ES=inner python3 cat_r145.py c_fix_inner
python3 seller_knn.py && EXTRAS=knn_title.pkl,sknn.pkl python3 cat_r145.py c_fix_sknn
python3 sknn_nested.py && EXTRAS=knn_title.pkl,sknn.pkl SKNN_NESTED=1 python3 cat_r145.py c_fix_sknn_nest
SEED=5151 python3 cat_r145.py c_fix_s2 && python3 avg_preds.py $AP2026Q2_DATA_DIR avg_fix2 b_fix_noso_b c_fix_s2
python3 eval_holdout.py $AP2026Q2_DATA_DIR /home/claude/work/holdout/holdout_labels.csv ../../results/round145_claude_holdout/metrics.json b_cat_noso_b b_fix_noso_b ...
```
Tiempos (1 hilo): sparse_all ≈ 3 min, nbsvm 40 s, knn_title 100 s, CatBoost 11–18 min por modelo (ver `summary.json`).
