# Ronda 36 — CatBoost de segunda etapa en vendedores nuevos

## Corrección metodológica

La ronda 35 mostró resultados opuestos de dos **arquitecturas distintas** de
CatBoost: segunda etapa sobre predicciones OOF en cortes aleatorios, y modelo
nativo sin `seller_id` en cortes por vendedor. Ese contraste no podía aislar
el efecto del corte. Aquí se reentrenó CatBoost de **segunda etapa**, con los
mismos predictores de texto word/char OOF y variables profundas que HistGB de
ronda 30, en los tres cortes seller-group 883, 2026 y 314159. Los folds
internos de texto y codificaciones se agrupan por vendedor. El texto es el
`docs` original del Google Sheet de 100.000 filas.

## Validación pareada

Control: HistGB de ronda 30, umbral 0,475. CatBoost: umbral 0,5. La mezcla
50/50 usa 0,4875, congelado en ronda 30. Todos los holdouts están compuestos
por vendedores ausentes del entrenamiento.

| Corte | Filas | HistGB correctos | CatBoost arreglos/roturas (neto) | Mezcla 50/50 arreglos/roturas (neto) |
|---|---:|---:|---:|---:|
| 883 | 14.335 | 13.314 | 102/94 (+8) | 38/37 (+1) |
| 2026 | 13.493 | 12.390 | 105/109 (−4) | 55/53 (+2) |
| 314159 | 13.898 | 12.758 | 146/114 (+32) | 67/56 (+11) |

En la corrección unilateral hacia `new`, CatBoost obtiene +2, −8 y +10.
Para CatBoost completo, los intervalos bootstrap 95% del neto, agrupados por
vendedor, son aproximadamente [−19,+36], [−32,+26] y [−2,+69]. Los tres
incluyen cero, y los cortes reutilizan train y pueden solaparse. La tercera
semilla ya se había usado para estudiar otras variantes: no es un confirm
independiente. La mezcla fija 50/50 mejora levemente en los tres, pero sus
intervalos también incluyen cero. No hay una ganancia de accuracy confirmada
que justifique una submission.

## Refit full70k y comparación con la 33

Se ajustó la misma arquitectura con 70.000 etiquetas y se guardaron
probabilidades para 30.000 avisos de test, alineadas por `kaggle_ID=1..30000`.
El CatBoost cambia 431 decisiones frente al HistGB de ronda 30; la mezcla
50/50 cambia 215. Frente al CSV exacto de la submission 33, discrepan en
1.280 y 1.288 filas respectivamente, de las cuales 327 y 325 son de
vendedores nuevos. Esas discrepancias **no** son aciertos medidos. Como la
validación anterior no demuestra mejora sobre HistGB y tampoco reproduce la
regla completa de la 33, no se generó CSV de submission.

## Decisión y siguiente prueba

No promover el CatBoost aislado ni su mezcla. El resultado sí acota un paso
útil: una futura candidata debe compararse contra un proxy reproducible de la
33 en **vistos** y **nuevos**, y reservar un confirm que no haya influido en
el diseño de la regla. La tasa de vendedores vistos de test es 21.487/30.000;
un holdout exclusivamente de nuevos no basta para evaluar la mejora total.
Las pseudoetiquetas de test y scores públicos no se usaron para entrenar ni
validar este modelo.

## Reproducción

Con Python 3.12, catboost 1.2.10, scikit-learn 1.8.0, numpy 2.3.5,
pandas 2.2.3 y scipy 1.17.0, desde la raíz del checkpoint:

```
python work20/round36_group_secondstage_cat.py 883
python work20/round36_group_secondstage_cat.py 2026
python work20/round36_group_secondstage_cat.py 314159
python work20/round36_full_secondstage_cat.py
```

El checkpoint incluye fuentes de features y del modelo, datos originales,
probabilidades pareadas, probabilidades full70k/test, informes y SHA256.
