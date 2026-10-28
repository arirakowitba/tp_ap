# Ronda 51: arbitraje CatBoost en desacuerdos de texto

Hipótesis: cuando HistGB y FastText discrepan, la coincidencia de CatBoost con FastText identifica un segmento de mayor transferencia a vendedores nuevos. CatBoost se ajustó solo en fit de cada split seller-group; se evaluó confirm con etiquetas reales de train. Umbrales fijos: HistGB/FastText .475 y CatBoost .5.

| Confirm | Casos CatBoost = FastText ≠ HistGB | Fixes/breaks frente a HistGB | Neto | IC 95% por vendedor |
| --- | ---: | ---: | ---: | ---: |
| 1 | 91 | 55/36 | +19 | [0,+37] |
| 2 | 114 | 74/40 | +34 | [+12,+56] |
| 3 | 98 | 62/36 | +26 | [+7,+45] |

Los cortes comparten el mismo train y se solapan; los intervalos no deben combinarse como tres réplicas independientes. En test, con modelos completos, hay 81 desacuerdos de vendedores nuevos en que CatBoost coincide con FastText. Solo 20 implicarían cambiar la submission 41: 15 `used→new` y 5 `new→used`. De esos 20, doce ya aparecieron en el bloque de 34 cambios de la submission 43, que empató públicamente con 33; otros tres (7089, 24522 y 26141) habían sido excluidos de pruebas previas. El empate de 43 informa al **bloque agregado**, nunca etiqueta a esas doce filas.

Conclusión: hay una señal confirmada frente a HistGB en este tipo de desacuerdo, pero todavía no justifica una submission sobre 41. La arquitectura CatBoost es un proxy, no reproduce la política manual 41, y la superposición con la 43 hace especialmente incierta la transferencia pública. Priorizar una regla que supere una segunda base en confirm o identifique una familia de cambios realmente nueva. No se usaron etiquetas de test ni se envió submission.

`round51_cat_arbitration.py` reproduce los segmentos y el diagnóstico de test desde los pares CatBoost de ronda 50 y las probabilidades completas existentes. `round51_cat_arbitration_report.json` conserva los 20 IDs solo como **desacuerdos de modelos**, no como etiquetas inferidas.
