# Ronda 38 — DART sobre texto OOF y variables profundas

## Hipótesis y protocolo

Se probó un modelo de segunda etapa con *tree dropout* (`LightGBM` DART,
700 árboles, 23 hojas, `drop_rate=0.12`) para reducir dependencia de árboles
individuales al transferir a vendedores nuevos. Recibió exactamente las
variables numéricas, codificaciones y predicciones word/char OOF agrupadas
por vendedor del HistGB de ronda 30. `docs` es el texto original del Sheet
de 100.000 filas. Se reservaron vendedores enteros con semillas 883 y 2026.
La decisión `used` usa 0,475 para ambos modelos, sin optimización posterior.

| Corte | Filas | HistGB correctos | DART correctos | Arreglos | Roturas | Neto |
|---|---:|---:|---:|---:|---:|---:|
| 883 | 14.335 | 13.314 | 13.274 | 92 | 132 | −40 |
| 2026 | 13.493 | 12.390 | 12.362 | 84 | 112 | −28 |

El *log loss* de DART también empeoró frente a HistGB: 0,195464 contra
0,183874 y 0,218283 contra 0,213349. Una mezcla diagnóstica de 25% DART
y 75% HistGB obtuvo −7 y +5 aciertos: tampoco es estable. No se ensayó una
tercera semilla ni se ajustaron umbral o peso después de ver los holdouts.

**Decisión:** descartar esta configuración y no hacer full70k ni submission.
El resultado solo cubre este DART y estos features; no prueba que cualquier
modelo con dropout o cualquier LightGBM sea inferior.

## Política de la 33

Se confirmó que los 31 cambios exactos de la 33 sobre la 31 incluyen 19
`new→used` y 12 `used→new`; 19 pertenecen a vendedores nuevos y 12 a vistos.
Los modelos guardados no permiten atribuirlos a una única regla simple de
dirección y segmento. El CSV exacto de la 33 es una base válida para comparar
predicciones de test, pero sus reglas por ID no se pueden trasladar a train
como si fueran una política validada. Ninguna inferencia de score público se
convirtió en etiqueta individual.

## Reproducción

Con Python 3.12, lightgbm 4.6.0, scikit-learn 1.8.0, numpy 2.3.5,
pandas 2.2.3 y scipy 1.17.0:

```
python work20/round38_group_dart.py 883
python work20/round38_group_dart.py 2026
```

El checkpoint contiene fuente, datos originales, predicciones pareadas,
CSV 31/33, resultados y manifiesto SHA256.
