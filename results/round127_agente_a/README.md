# Ronda 127 — Agente A: información nueva en las features (28-10-2026)

Protocolo igual al de la ronda 126: 5 folds `StratifiedKFold(shuffle=True, random_state=2026)` sobre los 70.000 de train; toda codificación con etiqueta es OOF. Un solo hilo siempre.
Código en `src/round127_agente_a/`:
- `build_feats_a.py` arma tres tablas (100.000 filas, orden de `feats.pkl`): `a_feats_raw.pkl` (campos crudos del JSON), `a_feats_sel.pkl` (agregados de vendedor sobre train+test, sin etiqueta), `a_feats_cat.pkl` (contexto de categoría y TE OOF de categoría × tipo de aviso/cantidad/modo/precio).
- `train_a.py` = copia de `train_gbm.py` con las mismas TE, stacks y agregados, más `AFEATS=...` para sumar tablas. Por defecto CatBoost profundidad 5, lr 0,06, corte por logloss.

## Revisión de campos crudos (sin modelo)

- `listing_source`, `international_delivery_mode`, `coverage_areas`, `differential_pricing`, `subtitle`, `site_id`: constantes o vacíos.
- `descriptions`: el id es `MLA<item>-<n>` y `n` sigue a `date_created` (corr. 0,985): no aporta nada nuevo.
- `id` / `parent_item_id`: números sin relación con la etiqueta ni con la fecha (ofuscados). Tener padre: 47 % vs 45 % de usados, nada.
- `shipping.methods` vacío siempre; `dimensions` casi siempre nulo.
- Fotos: el sufijo del id trae mes/año de subida (`_MMYYYY`); el desfase foto→publicación casi no separa (42–47 % de usados por tramo).
- Atributos: kilómetros de autos (`-KMTS`): con km > 100 es casi siempre usado (717 de 719), con km 0 es nuevo. Pocas filas (≈700).
- Permalink vs título: en el 1,2 % el título cambió después de publicar (66 % usados vs 46 %).

## Experimentos

Proxy: LightGBM 63 hojas, lr 0,05, 1 hilo (≈1,5 min), mismos stacks (`sparse_all`, `nbsvm`) y `knn_title.pkl`.

| Nombre | Idea | OOF acc | logloss | vs base | Veredicto |
|---|---|---:|---:|---|---|
| `a_lgb_base` | proxy sin features nuevas | 0,94169 | 0,14769 | — | referencia del proxy |
| `a_lgb_raw` | + campos crudos | 0,94201 | 0,14595 | +0,03 pt / −0,0017 | ayuda en logloss |
| `a_lgb_sel` | + agregados de vendedor | 0,94171 | 0,14765 | 0 / 0 | nada solo |
| `a_lgb_cat` | + contexto de categoría | 0,94197 | 0,14727 | +0,03 / −0,0004 | poco |
| `a_lgb_all` | + las tres | 0,94316 | 0,14519 | +0,15 / −0,0025 | prometedor → CatBoost |
| `a_cat_all_s81` | CatBoost d5 lr .06 con las tres tablas (5 folds, seed 8101) | 0,94969 | 0,13000 | peor que todos los `cat_d5_ll` (0,9506–0,9513 / 0,1254–0,1273) | **empeora**: demasiadas features nuevas; se separa por grupo |

Proxy 2: CatBoost d5 lr 0,15, solo folds 0 y 2, seed 8201 (≈8 min). Más fiel a CatBoost que LightGBM.

| Nombre | Idea | acc (f0+f2) | logloss | vs `a_cq_base` | Veredicto |
|---|---|---:|---:|---|---|
| `a_cq_base` | sin features nuevas | 0,94946 | 0,12833 | — | referencia |
| `a_cq_raw` | + campos crudos | 0,95111 | 0,12536 | +0,17 pt / −0,0030 | **mejora** → CatBoost completo |
| `a_cq_sel` | + agregados de vendedor | 0,94950 | 0,13018 | 0 / +0,0019 | empeora |
| `a_cq_cat` | + contexto de categoría | 0,94882 | 0,13388 | −0,06 / +0,0056 | empeora (fold 2 se rompe: ll 0,141) |

Confirmación con CatBoost completo (5 folds, d5, lr 0,06, corte por logloss):

| Nombre | Idea | OOF acc | logloss | vs base | Veredicto |
|---|---|---:|---:|---|---|
| `a_cat_raw_s81` | + `a_feats_raw.pkl` (seed 8101) | 0,95159 | 0,12538 | mejor single: acc arriba del rango 0,9506–0,9513, logloss igual que el mejor (0,1254 de `lr03b`, lr 0,03) y mejor que todos los lr 0,06 (0,1261–0,1273) | **mejora modesta**, falta 2.ª semilla |
| `ens_ll8` + `a_cat_raw_s81` (peso 0,3) | mezcla | 0,95197 | 0,12364 | ens_ll8: 0,95187 / 0,12441 | baja logloss 0,0008; corr. con ens 0,9969 (igual que entre semillas) |

Ideas del coordinador (proxy CatBoost lr 0,15, folds 0+2, seed 8201; referencia `a_cq_raw` 0,95111 / 0,12536):

| Nombre | Idea | acc (f0+f2) | logloss | vs `a_cq_raw` | Veredicto |
|---|---|---:|---:|---|---|
| `a_cq_raw_mat` | + "madurez" del vendedor (`a_feats_mat.pkl`: orden del aviso en el vendedor por fecha, avisos el mismo día, span de fechas, títulos duplicados, n en test) | 0,95014 | 0,12704 | −0,10 / +0,0017 | empeora |
| `a_cq_raw_ctr` | CTR CatBoost `Borders:Prior=0.5/1 + Counter`, `counter_calc_method=Full`, `max_ctr_complexity=4` | 0,94929 | 0,13171 | −0,18 / +0,0064 | empeora claro |
| — | `parent_item_id` | — | — | sin señal (47 % vs 45 % usados, número sin relación con fecha) | descartado sin modelo |
| `a_cat_raw_lr03_s83` | + `a_feats_raw.pkl`, lr 0,03 (seed 8301) | 0,95053 | 0,12725 | peor que los 3 `cat_d5_ll_lr03*` (0,9511–0,9513 / 0,1254–0,1260) | **no confirma**: la mejora de `s81` era ruido de semilla |
| `a_sparse_raw` | stack lineal: título palabra + garantía + tokens crudos (atributo=valor, variaciones, tags, tamaños de foto, envío) + one-hot de `F`+`raw` binneados (`sparse_raw.py`, sin char n-gramas por memoria) | 0,92474 | 0,19413 | `sparse_all` 0,9273 / 0,1894; corr. 0,990 | aporta poco: logística sobre logits sa+nb 0,1831 → sa+nb+raw 0,1809 |
