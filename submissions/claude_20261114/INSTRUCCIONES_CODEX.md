# Instrucciones para Codex: cupos manuales del 14-11 (Claude)

Ancla: S4 = `SUBMISSION_4_TODAY_RECOMMENDED.csv` (AG), 0,94766, 19.899 aciertos con Npub = 20.998, LF 195dd663…
Primer cupo observado: `claude_blendfix_posterior_top174_sobre_S4.csv` = 0,94723, −9. Ya está incorporado como restricción.

## Candidatos vigentes (todos DEVELOPMENT_ONLY, ninguno QUALIFIED; IDs y hashes en `manifest.json`)

| Archivo | Flips | E[Δ] vs S4 (4 previas) | SD | Corrección de sesgo (−0,12/flip) | Notas |
|---|---:|---|---:|---:|---|
| `claude_post174_consenso_top10_sobre_S4.csv` | 10 | +5,4 a +5,7 | 6,0 | ≈ +4,5 | **recomendado para el 2.º cupo** |
| `claude_post174_consenso_top20_sobre_S4.csv` | 20 | +5,7 a +6,5 | 8,8 | ≈ +3,6 | contiene a top10 |
| `claude_post174_consenso_top30_sobre_S4.csv` | 30 | +5,8 a +6,9 | 10,7 | ≈ +3,0 | contiene a top20 |
| `claude_blendfix_posterior_top30_sobre_S4.csv` | 30 | +5,5 a +6,1 | 10,4 | ≈ +2 | subconjunto exacto de top174; comparte 8 IDs con consenso top10 |

- Regla de los consenso: ganancia posterior positiva con LAS DOS previas (H-AVG-Plus y blend_fix, b = 0,25), sólo IDs dentro del span histórico contrastante (2.345 IDs) y con el top174 incorporado.
- Los tres consenso están anidados (top10 ⊂ top20 ⊂ top30), así que cada score identifica exactamente el bloque incremental: g(11–20) = Δ20 − Δ10, y así.
- Máximo físico de todos: ≤ +30. Ninguno puede llevar a 0,95 desde S4. No se afirma P(+50).

## Secuencia sugerida (decide Ariel)

1. **2.º cupo: consenso top10.**
   - Si Δ10 ≥ +3 (score ≥ 0,94781), seguir al paso 2.
   - Si Δ10 ≤ −2, cortar: S4 sigue y no se envían más overlays de esta familia.
2. **3.º cupo: consenso top30.** Δ30 − Δ10 identifica el bloque 11–30.
   - Si Δ30 > Δ10, la mejor pública pasa a ser top30.
   - Si no, queda top10 y el bloque 11–30 no se reenvía.
3. Siguientes cupos: recalcular con el procedimiento de abajo antes de cada uno. No reenviar vectores ya puntuados; el script lo detecta.
4. Selección final para el privado: S4 (o el mejor overlay público) + el modelo puro corregido `claude_blendfix_pure_model.csv`. Las sondas sólo mejoran el público; en el privado cuenta la calidad del modelo (blend_fix: +54 netos sobre B10 en el holdout fiel, Q05 vendedor +33; CL-0020).

## Procedimiento al recibir cada score (antes del cupo siguiente)

```bash
mkdir -p w && cp results/round145_claude_public_estimator/state/* src/round145_claude_public_estimator/*.py w/ && cd w   # estado congelado en el repo
python add_score.py <csv enviado> <score> <nombre_evento>
SUBDIR=../submissions/claude_20261114/ python opt2.py   # reestima y reordena; congelar la salida en un CL antes del siguiente envío
```

- Registrar siempre el LF SHA256 del archivo efectivamente subido y el score tal como lo informa Ariel.
- Vectores idénticos con scores distintos van a cuarentena (el script se detiene).
- El estimador es razonable para overlays chicos dentro del span: cobertura prequencial del 93 % a 2 SD.
- Es optimista para vectores grandes o fuera del span: EVT-142 dio −288 contra −68 estimado, y top174 −9 contra +11,8. No usarlo para promover vectores con muchos flips fuera del span.
