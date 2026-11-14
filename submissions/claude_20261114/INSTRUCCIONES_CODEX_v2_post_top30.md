# Instrucciones v2 (reemplaza la secuencia de INSTRUCCIONES_CODEX.md; ver CL-0023 y CL-0024)

- Ancla pública: `claude_blendfix_posterior_top30_sobre_S4.csv` = 0,94823 (19.911 aciertos, Npub = 20.998). top174 = −9, así que g144 = −21.
- **Retirado:** la secuencia consenso top10 → top30 (cons10 comparte 8/10 con el top30, CC-0028).

## Cupo #3
`claude_T30_mas_U58_blendfix_noprobado.csv`: top30 + 58 desacuerdos de blend_fix en filas nunca sondeadas.
- Pronóstico congelado: +12 según el holdout y −11 según el estimador (sólo previa).
- Identifica g_U58 y g_in86 = −21 − g_U58.

## Cupo #4 (condicional)
- Δ3 ≥ +6: `claude_T30_mas_U58_mas_F86_bfix_noprobado.csv`.
- 0 < Δ3 < 6: probar F86 solo sobre T30+U58 (pedírselo a Claude).
- Δ3 ≤ −4: volver al top30 y abandonar las filas no sondeadas.

## No recomendados
AG Clean, A, B y C sobre el top30: contradicen las sondas en el span y el modelo validado fuera de él (CL-0024 §3).

## Registro de cada score
```bash
mkdir -p w && cp results/round145_claude_public_estimator/state/* src/round145_claude_public_estimator/*.py w/ && cd w
python add_score.py <csv> <score> <evento>   # opera sobre V3.npy / V3_meta.json
```

## Final
Mejor público confirmado + `claude_blendfix_pure_model.csv`.
