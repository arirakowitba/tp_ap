# Ronda 127 — Agente B: otros learners, texto y ensamble (28-10-2026)

Mismo protocolo que la ronda 126: 5 folds estratificados (semilla 2026) sobre los 70.000 de train, todo OOF.
Base: `ens_ll8` (promedio de 8 CatBoost prof. 5) **0,9519 / logloss 0,1244**. Diferencias de accuracy < 0,1 punto son ruido de semilla.
Código: `src/round127_agente_b/` (con `AP2026Q2_DATA_DIR=/home/claude/data`). Predicciones: `pred_b_<nombre>.npz` (`oof` orden train, `te` orden ID).

| Experimento | Idea | OOF acc | logloss | vs ens_ll8 | Veredicto |
|---|---|---:|---:|---|---|
| `b_lgb_t1` | LightGBM 31 hojas, l2 10, sin `seller_id`/`category_id`/`city` crudos (sólo TE OOF), misma tabla que CatBoost | 0,9438 | 0,1432 | −0,8 pts solo | más débil que CatBoost, pero útil en el apilado |
| stack `ens_ll8` + `b_lgb_t1` | logística sobre logits, CV anidada en los mismos folds | 0,9533 | 0,1217 | +0,14 / −0,0027 | **mejora**: peso negativo sobre LightGBM (−0,6) |
