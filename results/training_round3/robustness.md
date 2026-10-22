# Robustez de la contribución de texto y los ensambles

## Qué se repitió y qué no

Las dos particiones se construyeron sobre las mismas 70.000 filas. La segunda reentrenó sus bases con otra semilla y cerró los hiperparámetros antes de confirmar. Es una comprobación de sensibilidad a la partición, no un holdout históricamente virgen. Los conjuntos de confirmación comparten **2.909 filas y 3.594 vendedores**; además, sus entrenamientos comparten gran parte de los datos. No sumar resultados, p-values o tamaños muestrales como si fueran pruebas independientes.

| Partición / modelo | Accuracy global | Vistos | Nuevos | Log loss | Corrige / rompe frente nativo |
|---|---:|---:|---:|---:|---:|
| Primera: nativo | 93,157% | 95,845% | 87,047% | 0,16996 | Referencia |
| Primera: 85% nativo + 15% NB | 93,350% | 95,886% | 87,585% | 0,17633 | 73 / 46: **+27** |
| Primera: meta de cinco modelos | 93,507% | 95,752% | 88,403% | 0,16102 | 214 / 165: **+49** |
| Segunda: nativo | 92,679% | 95,603% | 86,105% | 0,17725 | Referencia |
| Segunda: 85% nativo + 15% NB | 92,943% | 95,676% | 86,801% | 0,18162 | 66 / 29: **+37** |
| Segunda: meta de dos modelos | 93,243% | 95,665% | 87,799% | 0,16790 | 191 / 112: **+79** |

La mezcla fija 85/15 es el diseño comparable entre particiones. Su aporte positivo se repite, especialmente en vendedores nuevos. Los dos metas tienen distintas entradas y procedimientos: el primero usa cinco modelos e interacciones con C elegido en tune; el segundo usa nativo y NB con C=0,1 fijo. No constituyen una réplica exacta del mismo meta.

La caída del score absoluto del nativo en la segunda partición no implica que el proceso haya empeorado: se evalúan filas diferentes. Tampoco se debe comparar el 93,243% de esa partición con el 94% histórico sobre 4.147 filas distintas para decidir cuál modelo es mejor.

## Dónde aparece el aporte

- La mezcla fija suma **23 y 30** aciertos netos en vendedores nuevos. De ellos, **20 y 29** corresponden a publicaciones de cantidad inicial igual o menor que uno.
- En vendedores vistos, la misma mezcla suma solo **4 y 7** aciertos netos. El meta de cinco modelos pierde nueve en vistos y gana 58 en nuevos: hay una compensación real que debe explicitarse.
- El segmento nuevo vendedor + cantidad uno tiene accuracy base de **83,65% y 82,12%**. Los respectivos metas lo elevan a **85,28% y 84,44%**. Sigue siendo el cuello de botella principal.
- El aporte por categoría es disperso y abundan categorías con pocas filas. No hay evidencia para activar o invertir una política completa por categoría. Como diagnóstico, MLA2044 pierde dos y seis casos netos con los metas; esto justifica inspección semántica, no una regla automática.
- En las filas donde NB y nativo discrepan, el nativo todavía acierta **76,89% y 73,64%**. Un flip sistemático hacia NB sería claramente perjudicial. La señal útil está en ponderar el desacuerdo, no en su mera existencia.

## Incertidumbre condicional

Un bootstrap por vendedor, con 1.500 repeticiones, estima los siguientes intervalos del cambio de accuracy de la mezcla fija frente a su nativo:

| Partición | Delta observado | IC95% por vendedor |
|---|---:|---:|
| Primera | +0,193 puntos porcentuales | [+0,046; +0,333] pp |
| Segunda | +0,264 puntos porcentuales | [+0,130; +0,406] pp |

Estos intervalos son condicionales a los modelos ya ajustados. No incluyen toda la incertidumbre debida a selección de arquitectura, búsquedas previas, reutilización del dataset o cambio de distribución. No acreditan una probabilidad del 95% de mejorar Kaggle ni una accuracy global final de 96%.

## Frente a la referencia histórica más fuerte

La mejora del meta histórico coherente es pequeña: 94,357% frente a 93,972% en las mismas 4.147 filas, con umbral histórico 0,505. El bootstrap por vendedor para ese cambio incluye cero: **[-0,074; +0,817] pp**. También incluyen cero los intervalos de mejora de Brier y log loss. Es un candidato prometedor y no una superioridad demostrada.

La calibration audit encuentra una distinción importante:

| Métrica sobre 4.147 filas | Histórico | Meta coherente |
|---|---:|---:|
| Brier | 0,04538 | 0,04370 |
| Log loss | 0,15506 | 0,15096 |
| ECE de diez bins | 0,00994 | 0,01510 |
| ECE en vendedores nuevos | 0,01929 | 0,03485 |

Brier y log loss mejoran, pero ECE empeora. No afirmar que la calibración mejora globalmente. ECE depende del binning y también tiene variación muestral; la conclusión operativa es revisar confiabilidad por segmento antes de transformar probabilidades en confianza para flips.

## Tres experimentos prioritarios

1. **Dropout del vendedor sobre la rama fuerte de profesor textual y variables profundas.** Ya está en ejecución. Mantener la ventaja del vendedor conocido y mejorar su fallback es más razonable que quitar seller_id globalmente. Comparar contra el mismo modelo sin dropout, no únicamente contra el nativo débil.
2. **Profesores de texto separados por campo y contexto de negación.** Un profesor para título y otro para garantía puede distinguir expresiones de estado de expresiones como “nuevo modelo”. Generar cada probabilidad OOF y cerrar sus pesos en tune. No diseñar reglas a partir de los resultados públicos de cohortes.
3. **Comprobación outer disjunta por vendedor del stack fuerte ya elegido.** Sirve para medir transferencia a vendedores realmente nuevos. No usar ese outer para otra búsqueda amplia; reportarlo por separado y usar la reponderación a la mezcla del test solo como sensibilidad.

## Implicación para futuros flips

Las probabilidades nuevas pueden ordenar casos dudosos y aportar otra fuente de evidencia. No convierten saldos agregados de cohortes en etiquetas individuales ni prueban pertenencia a la máscara pública. Un criterio de futuras pruebas debería exigir soporte calibrado fuera de muestra, desacuerdo entre familias y una hipótesis transferible; el consenso entre semillas correlacionadas o el voto del texto aislado no basta.

Los CSV de slices y la matriz JSON permiten auditar números sin volver a cargar todo el análisis. Ninguna etiqueta del test ni inferida mediante flips se utilizó para este informe.
