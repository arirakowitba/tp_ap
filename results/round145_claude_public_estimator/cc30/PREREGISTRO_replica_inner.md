# Preregistro de la réplica en el corte independiente (antes de ver su resultado)
Commit anterior a que terminara B10 en el corte interno (log en curso: iteración ~300 de la primera CatBoost).
- Corte: 56k del holdout → 44,8k fit / 11,2k evaluación (semilla 145, `make_inner_split.py`). Las filas de evaluación son disjuntas de las 14k del holdout externo.
- Modelos:
  - B10 = `b10_eval.py full` sobre el fit de 44,8k.
  - b_fix = `c_in_base` (cat_r145, STACKS sparse_fix,nbsvm, SEED 4242, LR 0,06), ya entrenado.
  - blend = 0,5·B10 + 0,5·b_fix.
- Reglas CONGELADAS e iguales a las del holdout de 14k:
  - U: blend ≠ B10.
  - F: b_fix ≠ B10, |p_bfix − 0,5| ≥ 0,05 y blend = B10.
  - Bandas: [0; 0,05), [0,05; 0,10), [0,10; 0,20), [0,20; 0,51].
  - Fallback a la banda cuando la celda tiene < 15 filas.
  - Regímenes respecto del fit de 44,8k.
- Métricas: n, fixes/breaks, neto, bootstrap por vendedor y por categoría (2.000 réplicas, semilla 42), neto por régimen, vistos/nuevos.
- Criterio de sostén, predeclarado:
  - U se sostiene si el neto en el corte interno es > 0 y su Q05 por vendedor ≥ 0.
  - F igual.
  - Si F no cumple, F86 queda descartado para los cupos.
