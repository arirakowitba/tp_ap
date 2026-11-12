# AP-2026-Q-2 · cartera adaptativa, reemplazo de AG-0037

Ancla pública S4/EVT-154, 0.94766. La meta exacta 0.95404 requiere 134 aciertos netos si Npub=20.998. Los 50 cupos fueron una referencia histórica. El presupuesto operativo se deja sin confirmar y los envíos están suspendidos.

## Secuencia de decisiones

| Familia | Reparto orientativo de un presupuesto de 50 | Evidencia para entrar a la cola |
|---|---:|---|
| Modelo o blend amplio | 20 | Comparación pareada con el mejor modelo train-only, dos regímenes de vendedor, auditoría de solapamiento con S4 y con los bloques ya negativos |
| Contraste cruzado | 15 | Una decisión nombrada que cambie según el score. Batches de casos preespecificados con control y matriz de pertenencia verificables |
| Híbrido disjunto | 10 | Dos máscaras disjuntas verificadas por ID sobre S4, más control observado o presupuestado para distinguir el efecto incremental |
| Explotación final | 5 | Reserva contingente que se materializa sólo cuando los resultados previos determinan un vector nuevo |

Estos pesos describen una cartera posible, no cupos reservados. En cada actualización se comparan la ganancia esperada del mejor score final tras la acción, la probabilidad de superar el ancla y el valor de información para la decisión siguiente. Si no hay calibración que se sostenga fuera de muestra, usar rangos de sensibilidad, marcar las probabilidades como no identificadas y priorizar preparar candidatos amplios con mejora local robusta. Comparar el valor del contraste con el costo de postergar ese candidato.

## Correcciones al diseño anterior

- Días 1–3. Retirar las 15 sondas individuales. Aun si los 16 cambios de D2 sumaran +1 sobre el control exacto, como máximo se recuperarían 7 aciertos públicos deshaciendo sólo los negativos. El límite se refiere a ese bloque y a ese control. Un score cero en un contraste de una fila puede indicar que la fila no pertenece al público.
- Día 4. Retirar la deducción del sexto cambio. La igualdad para restar cinco efectos del total exige verificar por ID que los seis cambios, con su dirección, sean exactamente el bloque del agregado original y que el control sea el mismo. 57/75 no tiene ese certificado en el manifiesto. Un nuevo diseño sólo define bloques cruzados con soporte, composición y control verificados, y observa agregados.
- Día 5. El manifiesto repite cinco veces el mismo SHA de D1+D2. Eliminar esas entradas. Usar una reserva de presupuesto que, cuando se autorice la fase de explotación, se llena con vectores nuevos y distintos del control, deduplicados por el mapa ordenado ID/target además del hash de bytes.

## Registro y calibración

Antes de cualquier score, registrar fecha de corte, acción, familia, control y hashes, soporte nuevo, desacuerdos, alternativas disponibles, predicción de ganancia y decisión por intervalos de resultado. Particionar cronológicamente, entrenar únicamente con los eventos anteriores y evaluar por bloques de día y familia. Comparar error absoluto y cobertura con predicción cero, controlar el redondeo a cinco decimales y la sensibilidad a Npub, y comprobar la selección prospectiva sin reoptimizar con resultados posteriores. Los candidatos locales de validación se mantienen separados de esta estimación pública.

El resultado anterior de −2 fuera de una proyección de +0,7 a +2,24 y los cinco bloques disjuntos que sumaron −302 frente a S4 son evidencia para exigir calibración por régimen y familia, no pseudoetiquetas ni un veto por ID. En híbridos disjuntos, la aditividad identifica efectos de bloques sólo para el mismo control y conjunto público. El export de los 106 envíos sigue pendiente para validar los IDs de evento del ledger.
