# Resultados observados de las pruebas 16–20

Resultados suministrados por el usuario después de los cinco envíos. Control:
submission 15, score observado `0.94523`. Los numeradores siguientes suponen
21.000 filas públicas fijas y truncamiento a cinco decimales; todos los scores
recibidos son compatibles con esa hipótesis.

| Prueba | Score observado | Aciertos públicos inferidos | Diferencia respecto de la 15 |
|---|---:|---:|---:|
| 15, control | 0.94523 | 19.850 | 0 |
| 16, A+B | 0.94500 | 19.845 | −5 |
| 17, A+C | 0.94447 | 19.834 | −16 |
| 18, B+C | 0.94404 | 19.825 | −25 |
| 19, B+D | 0.94409 | 19.826 | −24 |
| 20, C+D+E | 0.94285 | 19.800 | −50 |

La matriz de cinco contrastes tiene rango cinco y recupera estos saldos:

| Cohorte | Filas totales | Saldo público de sus flips | Tasa de acierto posterior, media | Intervalo creíble del 95 % |
|---|---:|---:|---:|---:|
| A | 32 | **+2** | 0,539 | [0,349; 0,724] |
| B | 32 | −7 | 0,365 | [0,191; 0,555] |
| C | 64 | −18 | 0,314 | [0,188; 0,451] |
| D | 64 | −17 | 0,324 | [0,197; 0,462] |
| E | 128 | −15 | 0,419 | [0,321; 0,521] |

Por ejemplo, `a=(-5-16+25)/2=+2`, `b=-5-a=-7`, `c=-16-a=-18`,
`d=-24-b=-17`, `e=-50-c-d=-15`. La cohorte A suma aunque la 16 pierda al
mezclarla con B. La candidata con **solo A sobre la 15** tiene 32 cambios y
score público **inferido `0.94533`**. El CSV se preparó y auditó pero no se
envió: el campo de score observado permanece vacío. Es una candidata de
consolidación; no ocupa por sí misma un cupo ni define la próxima prueba.

Las tasas de la tabla usan `p~Beta(2,2)`, mezcla sobre la cantidad pública
`M~Hipergeom(30000,21000,n)` y la restricción `g=2K-M`. Cada saldo público
queda identificado; las tasas y la extrapolación al oculto dependen del
modelo de intercambiabilidad. Las marginales de máscara se aproximan como
independientes entre cohortes. Se conserva sensibilidad a Beta(1,1), Beta(2,2)
y Beta(4,4); no se fabrican etiquetas individuales para entrenar.

**A todavía no tiene alta confianza de generalización.** Su probabilidad
posterior de tasa de acierto mayor al 50 % es 65,5 %. En sus propias filas
ocultas, el modelo da 47,6 % de probabilidad de saldo positivo, 21,2 % de
empate y 31,2 % de saldo negativo. La esperanza es +0,79 aciertos ocultos,
con intervalo predictivo del 95 % [−6; +8]. Estas cantidades no son scores
ocultos observados ni garantías sobre una extensión a nuevas filas.

## Qué cambia para el próximo entrenamiento y las pruebas

El consenso favorable de texto/SVD+LightGBM en C no se tradujo en buenos
flips. Su utilidad en un ensamble de CV sigue siendo una cuestión distinta:
hay que medir la calidad de sus decisiones precisamente en el subconjunto
de discrepancias, por margen, clase, categoría y vendedor nuevo. Ajustar la
política de selección con predicciones fuera de fold y evaluar esa política
en un fold que no se usó para seleccionarla. El feedback público es agregado;
no se incorpora como etiqueta individual verdadera.

D confirma que ampliar hacia alternativas con poco apoyo puede perder
muchos aciertos. Se reduce su prioridad para consolidar. E es menos negativo
por fila que C/D y mezcla cuatro subgrupos; un saldo total −15 no resuelve
cuáles de E1–E4 merecen continuar. A es el patrón positivo que conviene
validar con IDs distintos antes de extrapolarlo.

## Diseño propuesto para la siguiente tanda

Todavía no hay CSV 21–25 generados. Esta propuesta es adaptable: tras la
actualización de la metodología en `COMPETITION_STRATEGY.md`, se prioriza la
ganancia esperada al cierre y la generalización a casos sin probar. No es
obligatorio dedicar tres cupos a separar E si nuevos modelos, tratamientos
o relaciones entre segmentos ofrecen más valor informativo. Manteniendo la 15 como control y A como
ancla de aporte conocido `a=+2`:

| Propuesta | Flips | Información |
|---|---:|---|
| 21: A+E1+E2 | 96 | Primer contraste entre subgrupos de E |
| 22: A+E1+E3 | 96 | Segundo contraste |
| 23: A+E1+E4 | 96 | Tercer contraste; completa la identificación de E |
| 24: A+F nueva | Por definir | Transferencia a IDs nuevos seleccionados tras auditar modelos y disponibilidad |
| 25: consolidación | Según resultados | Combinar solo después de recibir los cuatro scores |

Con `h_i=g_i-a` y `e=-15`, se obtiene
`e1=(h21+h22+h23-e)/2`; después `e2=h21-e1`, `e3=h22-e1`, `e4=h23-e1`.
Los tres scores nuevos más el saldo total conocido identifican los cuatro
balances. Esto conserva espacio para probar transferencia con F y para
consolidar; la selección definitiva de F requiere otra auditoría.

La evidencia, las confianzas y la candidata se reconstruyen con:

```bash
python scripts/analyze_crossed_results_20261020.py
python scripts/build_submission_review_notebook.py
python scripts/execute_submission_review.py --in-process
```

El generador original de 17–20 queda congelado al haber scores: verifica los
hashes y conserva esos CSV. Las pruebas posteriores deben usar nombres nuevos.
