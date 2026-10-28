# Gate logístico sobre desacuerdos HistGB–FastText

Hipótesis: usar metadatos de título, precio, categoría, envío y publicación para decidir si adoptar FastText cuando discrepa de HistGB en vendedores nuevos. Entrenamiento del gate solo en los desacuerdos de **tune** con etiquetas reales; cuatro folds OOF agrupados por vendedor; confirm de vendedores distintos. Regresión logística L2 C=0.1. Umbrales 0.5, 0.65 y 0.75 predefinidos en código, sin selección posterior por confirm. La clase a adoptar se toma de la probabilidad FastText original; umbral base de `used` 0.475.

| Split | OOF tune, gate >0.65: neto/cambios | Confirm, gate >0.65: fixes/breaks/neto/cambios | Bootstrap 95% neto por vendedor | Log loss en desacuerdos confirm HistGB→gate |
|---|---:|---:|---:|---:|
| first 39002 | 0/40 | 16/7/+9/23 | [0,19] | .70361→.69258 |
| second 39011 | +4/42 | 30/13/+17/43 | [4,30] | .72967→.70343 |
| third 46077 | +6/32 | 23/9/+14/32 | [3,26] | .74440→.72287 |

Confirm por dirección en gate >0.65: first `used→new` +7/7, `new→used` +2/16; second +13/33 y +4/10; third +5/7 y +9/25. Margen FastText `abs(p_fast−.475)>=.05`: confirm first +9/23, second +16/34, third +14/32; margen menor: 0/0, +1/9, 0/0. Tune OOF dentro de margen menor: −2/2, −3/9, −2/2. Esto favorece auditar un filtro de margen, sin elegirlo a posteriori como confirmación.

**Límites decisivos.** Los tres splits usan los mismos 70.000 registros etiquetados, por lo que los confirm no son independientes entre sí ni de exploraciones anteriores. Se probaron tres umbrales, lo que hace optimistas los intervalos. Comparamos contra HistGB, **no** contra la política exacta de la submission 41. Solo 23–43 decisiones por confirm; los IC bootstrap son descriptivos. Los resultados de log loss son solo de las filas donde ambos modelos discrepan, no de toda la validación.

**Auditoría de transferencia a test sin generar CSV.** Entre 176 desacuerdos HistGB/FastText de vendedores nuevos en test, los tres gates entrenados en tune seleccionan 13, 36 y 13 a umbral .65 (en confirm seleccionaron 23/254, 43/293 y 32/273). La dirección test es respectivamente 6/7, 31/5 y 2/11 (`used→new`/`new→used`): gran inestabilidad. Dos de tres gates coinciden en 9 IDs, tres de tres en ninguno. Las decisiones seleccionadas difieren de la 41 en solo 6, 17 y 6 IDs; muchas propuestas de FastText ya coinciden con 41. Por los empates públicos de 42/43 y la diferencia de la política base, **no promover a submission**. El siguiente trabajo es una comparación pareada contra un proxy realmente reproducible de 41, o construir un modelo más amplio con una confirmación independiente de la selección.

No se emplearon etiquetas test ni scores públicos como etiquetas. Reproducción: `python3 agent_structured/gated_disagreement.py` desde la raíz del workspace con `data.zip`, `round39_disjoint_pairs*`, probabilidades full test de ronda 39 y dependencias numpy/pandas/sklearn. Archivos NPZ conservan índices, etiquetas de tune/confirm, probabilidades pareadas y predicciones gate. Test NPZ solo tiene probabilidades y IDs, no etiquetas.

Checkpoint: Library `libfile_b4fc33e6ecac819184d737fcf7dbb467` (`agent_structured_checkpoint_20261025.zip`). Código y pares se conservan allí.
