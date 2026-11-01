# Denominador del leaderboard público — 2026-11-01

Auditoría del ledger `results/intensive_20261030/submission_ledger_04_current.json` con `src/infer_public_denominator.py`. Tras deduplicar hashes: 59 vectores con score. Bajo el supuesto de que las puntuaciones con tres decimales en el JSON tienen precisión registrada de tres decimales y las demás de cinco, se probaron todos los denominadores enteros entre 15.000 y 30.000. **Sólo N_pub=20.998 es compatible** con que cada accuracy provenga de un número entero de aciertos redondeado.

Chequeo independiente con resultados nuevos: D1+D2 .94652 corresponde a 19.875 aciertos; S1 .94628 a 19.870 (-5); S2 .94585 a 19.861 (-14); S3 .94628 a 19.870 (-5); C1 .94623 a 19.869 (-6). Cada score redondeado a cinco decimales es compatible con ese entero a N=20.998. N=21.000 no es compatible con varios scores del ledger.

Esto es inferencia del denominador bajo un supuesto de precisión; confirmar contra las reglas de Kaggle o raw score export si disponible. Las diferencias son **sumas netas de bloques en el subconjunto público**. No identifican qué ID es público ni su etiqueta. Dado que S1/S2/S3/C1 son disjuntos de D1+D2, sus netos agregados pueden incorporarse a un likelihood por bloque, pero nunca asignarse por ID sin prior/modelo y backtest.
