# Resultado observado de la submission 10 — 18 de octubre de 2026

Registro del resultado recibido **después del corte** del notebook 23 y del snapshot `snapshot_20261018.json`. Este documento conserva el hallazgo para retomar los experimentos mañana; el notebook y el snapshot mantienen su corte anterior y deberán regenerarse juntos tras auditar el nuevo score.

| Campo | Valor |
|---|---|
| Archivo puntuado | `submission_10_lote64_ganancias_acumuladas.csv` |
| Score público mostrado | **0,94500** (94,500 %) |
| Evidencia | Captura suministrada por el usuario `image(20261019-011846).png`; filename y score visibles en la misma fila, estado Complete |
| Score público anterior (submission 07) | 0,94476 |
| Cambios del archivo 10 contra el 07 | 72: ocho rescates conocidos más 64 flips nuevos en cuatro bloques de 16 |
| Resultado de esos cambios | **+5 aciertos públicos netos** frente al 07 |
| Aporte de los ocho rescates disjuntos | **+2 aciertos netos**, inferido de resultados anteriores; se presupone control y archivo auditados sin cambios adicionales |
| Aporte neto de los 64 nuevos | **+3 aciertos públicos netos**, por sustracción del resultado observado y del rescate anterior |

## Reconciliación entera

El historial previo permitía dos hipótesis sobre el display del score: truncamiento a cinco decimales con **21.000** casos públicos o redondeo a cinco decimales con **20.998**. El nuevo 0,94500 **no** es compatible con un numerador entero sobre 20.998 y redondeo a cinco decimales. Al añadirlo a los 23 resultados anteriores y buscar tamaños enteros de público de 1.000 a 30.000, sólo queda **21.000 con truncamiento** entre esos dos mecanismos de display. Sigue siendo una inferencia sobre el mecanismo, no una especificación oficial comunicada por la competencia.

Con esa hipótesis: score 07 = **19.840/21.000**, score 10 = **19.845/21.000**. El control compuesto previo 07 + rescate de 06 suma **19.842/21.000**; el lote nuevo aporta **19.845 − 19.842 = +3**. El mejor público observado ahora es **94,500 %**; el control compuesto anterior no tuvo una medición propia, sólo una inferencia algebraica.

## Qué queda por averiguar

- Una suma de **+3 en 64 filas** no revela cuál alternativa es correcta ni cuántas de esas filas están en el subconjunto público. No asignar etiquetas individuales ni tratar las cuatro subcohortes de 16 como evaluadas por separado.
- Mantener los archivos 08 y 09 y el resto de pendientes como **no puntuados** hasta recibir evidencia; no sumar una mejora no observada.
- En la próxima ronda, ingresar el score y su evidencia al registro principal; recalcular el snapshot, el sistema conjunto de restricciones y las posteriores **sin duplicar** pruebas solapadas; volver a ejecutar el notebook. Revisar las hipótesis de visualización y todos los pasajes que aún mencionan dos mecanismos compatibles.
- Diseñar las próximas pruebas para separar los cuatro bloques de 16 o cruzarlos con lotes nuevos, comparando contra el mismo control auditado. Actualizar la selección tras cada score; conservar validación independiente antes de transferir conclusiones a la porción oculta.

El resultado confirma una ganancia neta agregada sobre el leaderboard público. No implica que las 64 correcciones individuales sean correctas ni cuantifica aún la mejora privada.
