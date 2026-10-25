# Ronda 35 — auditoría de transferencia por vendedor

## Pregunta

La corrección con CatBoost de segunda etapa sobre HistGB parecía mejorar
vendedores nuevos en los cortes aleatorios de ronda 13. Se la colocó junto a
las pruebas con CatBoost nativo sin `seller_id` de ronda 29, que retienen
vendedores completos fuera del entrenamiento. **Las dos variantes de CatBoost
no son el mismo modelo**: la tabla siguiente muestra una falta de portabilidad
entre arquitecturas y protocolos, no aísla el efecto del tipo de corte.
El análisis reproduce cada par con etiquetas de train, no con scores públicos.

## Resultado pareado

Se conserva el umbral 0,475 de HistGB y 0,5 de CatBoost. `H→C hacia new`
significa usar la variante de CatBoost indicada únicamente donde HistGB decide
`used` y aquella decide `new`.

| Validación | Segmento | Filas | Cambios H→C hacia new | Arreglos | Roturas | Neto |
|---|---|---:|---:|---:|---:|---:|
| Aleatoria tune, CatBoost segunda etapa | nuevo respecto del fit | 2.240 | 38 | 25 | 13 | +12 |
| Aleatoria confirm, CatBoost segunda etapa | nuevo respecto del fit | 4.311 | 51 | 31 | 20 | +11 |
| Vendedor completo 883, CatBoost nativo | nuevo | 14.335 | 359 | 156 | 203 | −47 |
| Vendedor completo 2026, CatBoost nativo | nuevo | 13.493 | 408 | 191 | 217 | −26 |
| Vendedor completo 314159, CatBoost nativo | nuevo | 13.898 | 389 | 151 | 238 | −87 |

La mejora aparente de los dos cortes aleatorios no supera individualmente una
prueba McNemar exacta bilateral al 5% (p=0,073 y p=0,161). Los tres cortes por
vendedor reutilizan el mismo train, por lo que no son réplicas independientes.
El cambio de signo **no** se puede atribuir al corte sin entrenar el mismo
CatBoost de segunda etapa bajo los cortes agrupados. Esa comparación se hizo
en la ronda 36; los números negativos de CatBoost nativo no refutan por sí
solos a la arquitectura de segunda etapa.

En vendedores vistos de los cortes aleatorios, el CatBoost completo pierde
7/4.760 y 13/9.689 aciertos frente a HistGB. En el test, 21.487/30.000
(71,6%) pertenecen a vendedores vistos en train, y 8.513 son nuevos. Los
flips de 37 y 38 se concentraban en vistos (140/146 y 195/201); en 39,
925/1.228 eran vistos. Esto muestra que una mejora comprobada solo en nuevos
no basta para sustentar una submission amplia. Los scores públicos negativos
de 37–39 no permiten asignar etiquetas individuales ni atribuir este patrón
como su causa única.

## Regla de validación para próximas candidatas

1. Mantener tune/confirm separados y predicciones pareadas para **vistos** en
   un corte similar a test; medir arreglos/roturas por dirección.
2. Validar la transferencia a **vendedores enteramente nuevos** con cortes
   agrupados, antes de ampliar cualquier regla descubierta en aleatorio.
3. Reportar el efecto total ponderado por la mezcla observada del test
   (71,6% vistos, 28,4% nuevos) solo como escenario descriptivo; no asumir
   que la fracción pública ni sus etiquetas tengan esa mezcla exacta.
4. Exigir ganancia en confirm para ambos segmentos, o una regla explícita que
   cambie solamente el segmento donde sobrevive. No seleccionar filas con
   scores agregados del leaderboard.

**Decisión:** no hay candidata nueva para submission en esta ronda. El pequeño
saldo positivo de validación aleatoria no basta para promover la corrección
CatBoost hacia `new`; hace falta la comparación de arquitectura emparejada en
cortes agrupados, realizada en la ronda 36.

## Reproducción

Con los archivos incluidos en el checkpoint:

```
python work20/round35_validation_shift_audit.py
```

El checkpoint contiene el código, predicciones HistGB y CatBoost pareadas,
IDs de los flips 36–41, etiquetas de train dentro de `data.zip`, resultados
y manifiesto SHA256. No contiene etiquetas de test ni envía submissions.
