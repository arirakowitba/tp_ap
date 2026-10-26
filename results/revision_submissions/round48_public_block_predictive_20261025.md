# Ronda 48: qué informan los empates públicos de 42 y 43

Scores comunicados por Ariel: 33 = 0.94614, 42 = 0.94614 (13 cambios), 43 = 0.94614 (34 cambios). La 43 contiene los 13 de 42; los otros 21 forman otro bloque. La 41 = 0.94628 sigue siendo la mejor observada. El empate de scores publicados no identifica el resultado de ningún aviso.

Se evaluó una comprobación predictiva de bloques usando únicamente etiquetas del confirm seller-group nuevo (seed 46077). Entre cambios `used→new`, la regla de margen ≥.05 tuvo 47 fixes y 34 breaks en 81 cambios; la de margen <.05, 43 fixes y 33 breaks en 76. Con prior Beta(1,1) para la tasa de fixes de cada grupo, se simularon 500.000 veces 13 y 21 cambios, suponiendo de forma **no verificada** que cada aviso entra independientemente en el público con probabilidad .7.

| Bloque | Neto público esperado bajo supuesto | Probabilidad de neto ≤0 | IC predictivo 95% de neto |
| --- | ---: | ---: | ---: |
| 42, 13 de margen alto | +1.43 | 38.1% | [−5,+7] |
| 43 menos 42, 21 de margen menor | +1.89 | 36.7% | [−6,+10] |

Por el tamaño pequeño de los bloques, resultados sin ganancia son compatibles con la validación, aunque ambos empates reducen el atractivo de seguir ampliando FastText. Esta simulación es diagnóstica: el tamaño público, selección de filas, dependencia por vendedor y exactitud tras redondear a cinco decimales son desconocidos. Los dos grupos y las validaciones comparten datos y modelo, por lo que **no se multiplican sus probabilidades** ni se usan scores públicos para actualizar etiquetas por ID. Tampoco se equipara HistGB con la política exacta de 33.

Decisión: mantener 41 como base pública y buscar una señal distinta. No se genera submission nueva desde estos empates. El script y los pares están en el checkpoint.
