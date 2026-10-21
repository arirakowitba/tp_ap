# Estrategia para 96 % global con margen y confianza explícita

Objetivo actualizado por el usuario: **96 % de aciertos en las 30.000 filas
completas**, usando conjuntamente entrenamiento, datos, inferencia posterior
y experimentos adaptativos hasta el cierre. Las 21.000 filas públicas y
9.000 ocultas siguen siendo una hipótesis de partición. Esta accuracy global
es el objetivo solicitado; no se afirma que sea la fórmula de la clasificación
final de Kaggle.

## La contabilidad del objetivo

Si `P` y `H` son las tasas reales de acierto pública y oculta:

\[
A_{global}=0.7P+0.3H,\qquad C_{global}\ge 28.800.
\]

Si los ajustes no transfieren al oculto y ese segmento se queda en un
**hipotético 94,523 %**, alcanzar 95 % público produce **94,8569 % global**.
Alcanzar 96 % público produce 95,5569 % global. Para 96 % global necesitamos
al menos 20.293 aciertos públicos: **96,6333 %**, unos 443 aciertos públicos
netos más que los 19.850 de la 15.

El 94,523 % procede del score de una submission ya ajustada con feedback;
**no es una medición independiente del modelo en el oculto**. La igualdad
entre las dos tasas de un modelo es un escenario de cálculo. La tabla
incluye también el mejor punto de CV disponible como otro escenario, sin
convertirlo en una cota oculta.

| Oculto supuesto | Público mínimo para 96 % global, aproximadamente |
|---|---:|
| 93,5857 % — punto de CV reutilizada | 97,0381 % |
| 94 % | 96,8571 % |
| 94,523 % — escenario de referencia | 96,6333 % |
| 95 % | 96,4286 % |
| 95,5 % | 96,2143 % |
| 96 % | 96 % |

Los valores enteros exactos y sus brechas están en `scenarios.csv`. La mejora
pública de la misma submission no informa por sí sola su accuracy oculta.
No conocemos a qué partición pertenece cada ID; toda política aplicada al
test puede modificar filas de ambas particiones.

## Qué significa tener margen y alta confianza

Propuesta operativa para evaluar candidatos:

- **Público objetivo: 97 %**, 20.370 aciertos; mejora neta de 520 frente a la 15.
- **Oculto central objetivo: 95 %**, para una media global de **96,4 %**.
- **Límite predictivo inferior del oculto: 94 %**, para una cota global de
  **96,1 %** con ese público. La condición mínima matemática sería 8.430
  aciertos ocultos, 93,6667 %, pero se busca margen adicional.
- **Criterio de aceptación:** `Pr(C_publico + C_oculto >= 28800 | evidencia)
  >= 0.95`, calculado con predicciones conjuntas calibradas y contrastado en
  validación. La media de 96 % por sí sola no satisface este requisito.

Actualmente **ese criterio de confianza no está establecido**. No tenemos
etiquetas ocultas ni una distribución predictiva conjunta validada que lo
certifique. No se reemplaza esa ausencia con un binomial de 9.000 filas
independientes o un descuento arbitrario del score público. El código
`evaluate_joint_hidden_draws` evalúa el criterio si se le entregan muestras
conjuntas válidas; no las inventa.

## Auditoría nueva de validación y distribución

Se auditó el archivo usado para agrupar los folds históricos:
**70.000 grupos distintos para 70.000 filas**. Por lo tanto, el cero de
solapamiento entre esos IDs **no prueba separación de vendedores**.
Las métricas OOF existentes siguen siendo las medidas obtenidas; la
descripción previa de folds separados por vendedor era incorrecta.

El test contiene 21.487 filas de vendedores presentes en entrenamiento
(71,6233 %) y 8.513 de vendedores nuevos (28,3767 %). Los folds de los modelos
complementarios tienen aproximadamente 67,4 % de validación con vendedor
visto. Los números exactos, métricas por segmento y una reponderación
ilustrativa a la mezcla del test están en `validation_distribution_audit.json`.
Esa reponderación supone estabilidad condicional y no certifica un score.

El ensamble actual obtuvo **96,1157 % OOF en vendedores vistos** y
**88,3605 % en vendedores nuevos**. Cada punto porcentual adicional en ese
segundo segmento representaría unos **85 aciertos más** en las 30.000 filas
si se transfiriera con la mezcla observada. Por eso el próximo entrenamiento
debe reforzar señales de contenido, categoría y contexto que no dependan
de haber visto al vendedor. Como escenario objetivo por segmentos,
98 % en vistos y 92,5 % en nuevos darían aproximadamente **96,4393 % global**.
Son metas de validación pendientes, no desempeños ya obtenidos ni cotas.

Protocolo a preparar antes de la siguiente búsqueda grande:

1. Una evaluación que refleje vendedores vistos, separando filas y familias
   de productos para evitar duplicados triviales; todos los ajustes y
   encodings se aprenden solo con el subconjunto de entrenamiento.
2. Una evaluación realmente separada por vendedor para el segmento nuevo,
   y diagnóstico de categorías nuevas, texto, clase y márgenes.
3. Reponderación a la mezcla medida del test, con estrés sobre cambios de
   composición y errores correlacionados por vendedor/categoría. Ajustar
   las políticas de ambos segmentos por separado si la evidencia lo permite.
4. Dentro de bloques externos de validación, simular un feedback público
   70 % y un 30 % sellado. Reproducir secuencias de probes agregados y evaluar
   las correcciones inferidas en el 30 % después de fijar las decisiones.
   Esta prueba debe medir el **procedimiento adaptativo completo**.

Los datos de entrenamiento y sus splits ya se reutilizaron: un split nuevo
no borra ese historial. Congelar el protocolo de confirmación y conservar
las decisiones previas permite interpretar su evidencia. Como referencia
metodológica, la [documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
explica la separación entre ajuste y evaluación, las transformaciones dentro
del fold y los splits por grupos. La mezcla de protocolos se propone a partir
de la distribución medida en este dataset.

## Cómo obtener cientos de aciertos y transferir lo aprendido

**Modelos y datos.** Mejorar la base completa y el ensamble: interacciones
de texto/atributos/garantía, regularización, representaciones y relaciones
de vendedor/categoría. Comparar políticas para vendedores vistos y nuevos,
umbrales y calibración por regímenes. El objetivo es elevar la predicción
de las 30.000 filas, incluyendo las que nunca participan en probes.
La prioridad del próximo entrenamiento es el segmento de vendedores nuevos,
sin perder la precisión del segmento conocido; no basta agregar peso a los
modelos de texto actuales, que por sí solos tampoco superan allí al ensamble.

**Política de corrección.** Aprender sobre filas OOF si un cambio favorecido
por una combinación concreta de señales corrige un error o introduce uno.
Usar márgenes, acuerdo entre familias, frecuencia de vendedor y categoría,
similitud y estabilidad de semillas. Ajustar y evaluar en particiones
distintas. Con los scores, actualizar la confianza de reglas y cohortes
mediante la verosimilitud de los saldos observados, conservando la dependencia
entre pruebas. Las etiquetas individuales inferidas no son verdad observada.

**Transferencia.** Aplicar una relación a casos nuevos cuando mejore en
validación, sobreviva a extensiones con IDs distintos y conserve margen bajo
escenarios adversos. Comprobar que también ayude en el segmento sellado de
las simulaciones. La inferencia sobre casos sin probar es parte central de
la estrategia; no hace falta identificar todos los flips posibles.

**Feedback actual.** A ganó +2 públicos en 32 casos, pero tiene confianza
limitada. B, C y D perdieron −7, −18 y −17; E perdió −15 en 128 y sigue
mezclando cuatro subgrupos. Reducir la prioridad de las reglas que fallaron,
sin eliminar su posible aporte dentro de otro ensamble o segmento. Separar
E solo si puede cambiar suficientemente la decisión final. Tres cupos para
resolver E no tienen prioridad automática frente a una nueva mejora de modelo.

Quedan **1.192 filas sin probar en el ranking histórico auditado**; no son
todos los casos inciertos que puede descubrir otro modelo. Para conseguir
520 aciertos públicos con solo esas filas, asignadas al público al azar,
haría falta una tasa media de acierto de flips cercana al **81,16 %**.
Esa calidad no está respaldada por A–E. Esta limitación cuantitativa obliga
a mejorar la base y las reglas, ampliar el conjunto de oportunidades y
seleccionar mejor, en vez de aumentar el lote indiscriminadamente.

Si una corrección acierta con probabilidad `q`, su saldo medio es `2q-1`.
Para un crédito exclusivamente público, un lote de `n` filas aporta en
esperanza `0.7*n*(2q-1)` aciertos públicos. Por ejemplo, con `q=0.75`, los
520 aciertos requieren alrededor de **1.486 flips**; con `q=0.80`, **1.239**.
Son presupuestos de esperanza, no intervalos de confianza. `flip_budget.csv`
reproduce esos cálculos.

## Próxima tanda y horizonte

Los primeros cuatro envíos deben comparar hipótesis que puedan alterar
muchas predicciones futuras: mejora de modelo/tratamiento, extensión a IDs
nuevos de una regla, política de corrección actualizada y un contraste de
segmento de alto valor (E u otro). Definir controles y solapamientos para
separar lo que aporta cada intervención. La quinta se construye al recibir
los cuatro scores. La evaluación de modelos sigue también fuera de Kaggle.

Tamaños orientativos 128–256, luego 512 o más solo si la calidad de selección
y la validación justifican la extensión. Un lote grande sin una regla mejor
no produce automáticamente información útil ni mayor ganancia esperada.

Para unas 75 evaluaciones en 15 días, una asignación inicial adaptable es:
24 de modelos/datos/políticas, 24 de relaciones entre segmentos, 12 de
confirmación con IDs nuevos y 15 consolidaciones. Los experimentos se
reordenan diariamente por su capacidad de mejorar la decisión final y su
confianza; no por cuántos casos dudosos alcancen. Verificar el cierre real
antes de convertir este horizonte aproximado en calendario fijo.

Reproducir las cuentas y la auditoría con:

```bash
python scripts/build_global_target_plan_20261020.py
python scripts/build_submission_review_notebook.py
python scripts/execute_submission_review.py --in-process
```

No se entrenó un modelo nuevo ni se presentó una submission durante esta
actualización de estrategia. `global_plan.json` registra los objetivos como
metas pendientes y el criterio de confianza como no demostrado.
