# Pruebas 17 a 20: medir, contrastar y explorar

**Estado 20/10/2026:** score de la 15 recibido: `0.94523`. La 16 y las
17–20 están preparadas; no se recibieron scores de ninguna de ellas. El
objetivo es aprender durante las próximas rondas para mejorar el score final,
incluido el 30 % oculto. Todas las pruebas comparan contra **el mismo CSV 15**;
no se deben sumar scores de dos pruebas distintas como si midieran filas
independientes. Conservar exactamente los CSV y hashes del manifiesto.

| Archivo | A (32) | B (32) | C (64) | D (64) | E (128) | Flips respecto de la 15 | Qué contrasta |
|---|---:|---:|---:|---:|---:|---:|---|
| 16, preparada anteriormente | 1 | 1 | 0 | 0 | 0 | 64 | Ancla A+B: votos altos frente a bajos |
| 17 | 1 | 0 | 1 | 0 | 0 | 96 | A frente a C, generalización del consenso |
| 18 | 0 | 1 | 1 | 0 | 0 | 96 | B frente a C, réplica cruzada |
| 19 | 0 | 1 | 0 | 1 | 0 | 96 | D: desacuerdo entre semillas y modelos |
| 20 | 0 | 0 | 1 | 1 | 1 | 256 | E: expansión a zonas menos exploradas |

Las 320 filas de A–E son distintas. A y B vienen de dividir la 16 por los
votos de siete semillas. C favorece el consenso de modelos CatBoost nuevos,
CatBoost anteriores y, si se terminó su CV agrupada, modelos de texto y
LightGBM con SVD. D incluye contradicciones entre semillas y modelos. E
contiene cuatro subgrupos auditados de 32: modelo nuevo, modelo anterior,
discordancia y extensión. En cada cohorte se equilibran las dos etiquetas
alternativas, se evitan vendedores repetidos y se limitan a cuatro los casos
de una misma categoría. Los pesos de selección son **rangos de prioridad**, no
probabilidades calibradas.

Se entrenaron dos modelos complementarios en tres folds agrupados por
vendedor: TF-IDF con regresión logística (`0.90933` accuracy OOF) y SVD de
texto más LightGBM (`0.91710`). El CatBoost anterior alcanza `0.93310` en
esos folds; combinar 70 % de ese CatBoost con 30 % de las señales nuevas
alcanza `0.93586` OOF. Son medidas internas reutilizadas para selección y
**no** predicen el resultado público ni oculto. La cohorte Q anterior perdió
19 aciertos públicos pese al apoyo del CatBoost anterior; por eso las
discrepancias entre familias son hipótesis para probar, no etiquetas seguras.
Ver `../../results/new_models_20261020/comparison.json`.

Sea `g_i = aciertos_publicos_i - aciertos_publicos_15`. Bajo el supuesto
de 21.000 filas públicas fijas, `0.94523` con truncamiento equivale a
19.850 aciertos. Una vez disponibles los cinco scores, la matriz de la tabla
tiene rango cinco y permite recuperar los saldos públicos **agregados**:

```
a = (g16 + g17 - g18)/2
b = (g16 - g17 + g18)/2
c = (g17 + g18 - g16)/2
d = g19 - b
e = g20 - c - d
```

Una mejora de C repetida en 17 y 18 refuerza que ese patrón aporta, aunque
las dos pruebas comparten filas: son contrastes, no réplicas independientes.
Si los valores no resultan enteros compatibles, revisar el denominador real,
modo de display, CSV y base de control. Un score público identifica balances
de cohortes, nunca la verdad de cada fila ni una ganancia garantizada en el
30 % oculto.

## Cómo decidir las siguientes rondas

1. Registrar cada score textual en
   `../../results/revision_submissions/feedback_iterativo_20261020.json`.
   Ejecutar `python scripts/execute_submission_review.py --in-process`.
   La sección 13 del notebook calcula el posterior de cohortes bajo el
   modelo de máscara pública y exporta una consolidación opcional con
   `actualizar_submission(salida=...)`. Comparar la recomendación con
   sensibilidades a otras previas y descuentos de transferencia al oculto.
2. Si A/B/C muestran ganancia consistente, buscar filas nuevas semejantes y
   probar su extensión contra la misma base documentada. Si D difiere del
   consenso, separar las fuentes de desacuerdo para evitar extrapolar una
   señal incorrecta. Si el agregado E sorprende, diseñar pruebas cruzadas
   adicionales sobre E1–E4: la prueba 20 **no** identifica sus cuatro efectos.
3. Reservar parte del presupuesto aproximado de 75 pruebas para terreno
   nuevo (textos y catálogo, vendedores/categorías menos representados,
   arquitectura complementaria), sin dedicar todo el presupuesto a explotar
   cohortes favorecidas por el leaderboard. Reentrenar y verificar la
   estabilidad por vendedores, folds y clases antes de extender un patrón.
4. Cerca del cierre, conservar la mejor submission observada y construir
   una candidata final con cohortes de ganancia pública identificada y
   transferencia al oculto plausible. Si se desconoce un score, dejarlo
   pendiente: la 16 y la 14 no tienen resultado suministrado.

El generador reproducible es
`../../scripts/compose_four_information_probes_20261020.py`; los archivos de
auditoría, hashes y lista exacta de IDs figuran en
`pruebas_17_a_20_manifiesto.json`.

## Presupuesto tentativo de 15 días

Si efectivamente quedan 15 días a cinco envíos por día, el máximo teórico
es 75 evaluaciones; el calendario y los cupos reales se revisan al recibir
cada resultado. Este presupuesto permite aprender sin agotar pronto los
casos más inciertos:

| Ventana | Envíos | Uso principal |
|---|---:|---|
| Días 1–3 | 15 | Matriz 16–20 (5); hasta diez contrastes para separar E y repetir patrones con IDs nuevos |
| Días 4–10 | 35 | 14 extensiones/validaciones de cohortes, 14 contrastes en zonas nuevas, 7 controles consolidados |
| Días 11–13 | 15 | 6 pruebas dirigidas, 6 contrastes nuevos, 3 candidatos consolidados |
| Días 14–15 | 10 | 4 confirmaciones en IDs independientes, 4 candidatos finales, 2 reservas según hallazgos |

Cada tanda se rediseña con los scores observados y la validación de modelos;
no se preasignan etiquetas a los 30.000 casos. Si la 16 todavía no se presentó,
subirla antes de interpretar la matriz 17–20. Con solo cuatro cupos disponibles
en esa situación, usar 16–19 y dejar la 20 para el siguiente cupo.
