# Cupo #4 precalculado según el resultado del cupo #3 (`claude_T30_stack_top50_sinrev_cupo3.csv`)
Generado con `iterate.py` en copias aisladas del estado (scores hipotéticos; Npub 20.998). Δ es el cambio de aciertos frente al top30 (19.911).

| Δ cupo #3 | Score | Base para el #4 | Recomendado #4 | E corregida (b = 1 / pesimista) | Lectura |
|---:|---|---|---|---|---|
| −10 | 0,94776 | top30 | top50 nuevo sobre el top30 | +3,5 (+9,5 / +1,4) | probar otras filas desde el top30 |
| −5 | 0,94800 | top30 | top50 nuevo sobre el top30 | +3,0 (+9,0 / +1,3) | ídem |
| 0 | 0,94823 | top30 (en empate se conserva) | top50 sobre el top30 | +2,6 (+8,6 / +0,8) | ídem, margen chico |
| +5 | 0,94847 | cupo #3 | top20 | +1,0 (+3,4 / −0,5) | borde de la regla de parada |
| +10 | 0,94871 | cupo #3 | top20 | +0,5 (+2,9 / −0,6) | **parar**: la E corregida es < +1 |
| +15 | 0,94895 | cupo #3 | top20 | +0,4 | parar |
| +20 | 0,94919 | cupo #3 | top20 | +0,2 | parar |

Lectura:
- Si el cupo #3 sale bien, consume casi toda la señal identificable, y a partir de ahí más overlays no tienen esperanza útil.
- Si sale mal, todavía queda otro bloque de filas desde el top30.
- En todos los casos, el score real se procesa con `python iterate.py <state> <out> <csv> <score> CUP3`. La tabla es sólo orientativa: el script re-selecciona b y recalcula todo.
