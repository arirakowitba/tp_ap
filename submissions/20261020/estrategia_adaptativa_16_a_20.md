> Actualizacion despues de recibir 16-20: las ganancias publicas de A/B/C/D/E son +2/-7/-18/-17/-15 bajo 21.000 filas publicas y truncamiento. Ver `../../results/revision_submissions/resultados_cruzados_16_a_20.md` y la seccion 13.4 del notebook. El contenido de este documento conserva la planificacion historica previa a esos resultados.

# Estrategia adaptativa: submissions 16 a 20

**Fecha:** 20/10/2026, Argentina. **Referencia:** submission 15, que el usuario informa que obtuvo buen rendimiento. Su score publico observado es 0,94523. La entrega 20 se construye después de observar las pruebas 16 a 19.

## Qué sabemos antes de gastar otro intento

- La submission 11 obtuvo 0,94519. Los 32 casos P agregaron cuatro aciertos públicos netos. La cohorte Q completa de 144 restó 19 frente a la 11. Son saldos agregados, no etiquetas individuales. La 15 tomó solo 16 casos seleccionados de los 48 Q4; el usuario informa que rindió bien, pero aquí todavía falta su score exacto.
- El conjunto de test tiene 30.000 filas. Si el 70 % público informado corresponde exactamente a una partición fija por filas, el denominador público es 21.000 y el oculto 9.000. Para un lote nuevo de 64 filas esperamos 44,8 públicas y 19,2 ocultas; el número real de filas públicas del lote no se conoce.
- Después de excluir P, Q, filas ya puntuadas o pendientes y filas que en la 15 ya fueron cambiadas, el ranking previo ofrecía **1.512 candidatos vírgenes**. La prueba 16 usa 64. Entre los 1.448 restantes, solo 41 tienen apoyo de por lo menos dos de las cinco familias de señales actuales; 133 tienen apoyo de una y 1.274 de ninguna. Por eso ampliar a cientos de flips *con las mismas señales* sacrificaría calidad. Hay que crear señales complementarias y comprobarlas con validación agrupada antes de diseñar los lotes siguientes.

## Cuatro pruebas, una quinta consolidación

| Intento | Diseño | Resultado que podemos identificar |
|---|---|---|
| **16: lote inicial** | Ya construido: **64 casos nuevos**, 32 hacia `new` y 32 hacia `used`, 64 vendedores distintos; 16 tienen tres apoyos y 48 tienen dos. Se cambia solamente este lote respecto de la 15. | Saldo público del lote 16, comparando exactamente 16 con 15. |
| **17: extensión condicionada** | Elegir un lote nuevo y disjunto después del score 16. Si 16 gana y la validación por familias respalda la dirección, buscar casos similares; si pierde, usar el contraste para rehacer el ranking. Orientación inicial: **48–96** flips. | Transferencia o fallo del patrón identificado en 16, sin confundirlo con sus 64 filas. |
| **18: contraste complementario** | Priorizar desacuerdos entre modelos antiguos, CV reciente y nuevas señales de texto/catálogo, con clases y vendedores diversificados. Usar una variante validada en folds agrupados y **un lote disjunto**; tamaño decidido tras 16 y 17. | Saldo de una familia de información distinta. |
| **19: ampliación informada** | Concentrar el lote más grande en las subcohortes respaldadas por 16–18 y CV; evitar repetir los mismos ID. Si ninguna familia demuestra ventaja, probar una subcohorte diagnóstica pequeña en lugar de forzar una gran caída de score. | Cobertura adicional y prueba de que la señal se mantiene al ampliar. |
| **20: consolidación** | Partir de la mejor referencia comprobada y combinar únicamente cohortes compatibles con evidencia pública positiva y soporte independiente; revisar interacciones, dirección de clase, categorías y estabilidad CV. Se genera **cuando estén los cuatro scores**. | Mejor predicción final sustentada por las pruebas del día. |

Cada prueba 16–19 compara un **solo conjunto nuevo disjunto** con la misma referencia 15, salvo que el contraste de un resultado aconseje cambiar explícitamente la referencia y documentarlo. Así `score_i - score_15` identifica el saldo de la cohorte `i`; comparar archivos que acumulan lotes sin controlar los anteriores confunde sus efectos. Se conserva la mejor submission puntuable aunque una prueba diagnóstica resulte peor.

## Lectura estadística del leaderboard

Sea `B` un lote de `n` flips, `M` la cantidad de esas filas presentes en el público, `K` el número de flips acertados entre esas `M`, y `G` el cambio neto en aciertos públicos respecto de la 15. Entonces

\[
M\sim\operatorname{Binomial}(n,0{,}70),\qquad G=K-(M-K)=2K-M,
\qquad \Delta\mathrm{accuracy}_{pub}=G/21000.
\]

Por ejemplo, ganar cinco aciertos mueve aproximadamente `5/21000 = 0,0002381` el score público. El valor mostrado con cinco decimales debe convertirse al **numerador entero compatible** antes de atribuir `G`, utilizando la hipótesis de truncamiento comprobada en las rondas anteriores. `G` no identifica `M`, `K` ni qué filas concretas acertaron.

Para cuantificar la incertidumbre de la tasa de aciertos de una cohorte, integrar sobre todos los valores factibles de `M` con `K=(M+G)/2` entero y `0≤K≤M`. Con una previa explícita `p~Beta(a,b)` sobre la tasa de aciertos de flips comparables:

\[
p\mid M,G\sim\operatorname{Beta}(a+K,b+M-K),
\quad
P(M\mid G)\propto\operatorname{Binomial}(M;n,0{,}70)
\frac{B(a+K,b+M-K)}{B(a,b)}\binom{M}{K}.
\]

La mezcla resultante y su predicción para las filas ocultas se usan como **apoyo para decidir cohortes**, con descuento por selección de pruebas y posible diferencia entre público y oculto. La previa se debe justificar con validación agrupada y sensibilidad a varias previas; no transformar una mejora del score público en supuestas etiquetas verdaderas de test.

## Ejecución y trazabilidad

La prueba 16 es `submission_16_frescos_consenso64_sobre15.csv`, generada por `../../scripts/compose_probe16_20261020.py`. El archivo `cohorte_16_auditoria.csv` contiene las 64 filas y sus señales; `prueba_16_manifiesto.json` guarda los hashes e ID. El CSV se validó con **30.000 filas**, columnas `ID,condition`, ID 1–30.000 en orden y **exactamente 64 diferencias** respecto de la 15. Se preservan las etiquetas verificadas. Para continuar, registrar los scores exactos de 15 y 16 y actualizar el ranking antes de construir la 17.
