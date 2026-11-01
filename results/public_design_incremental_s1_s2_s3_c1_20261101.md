# Ganancia de identificabilidad de las cuatro sondas del 31-10

Se recomputaron las firmas de intervención de los 59 vectores únicos en `submission_ledger_04_current.json` y se añadieron los mapas exactos de S1, S2, S3 y C1. S1–S3 se compararon con D1+D2 y C1 con S48; son 25+25+25+20 cambios. Se usaron sólo predicciones, no etiquetas de test.

| Medida | Antes | Después |
|---|---:|---:|
| Firmas únicas | 377 | 382 |
| ID jamás intervenidos | 27.662 | 27.568 |
| Firmas singleton | 215 | 216 |
| ID en firmas compartidas | 29.785 | 29.784 |

Las sondas alcanzan 94 ID nuevos, pero generan sólo cinco firmas adicionales y un único ID distinguible más por diseño. Haber probado bloques disjuntos no equivale a aprender una etiqueta para cada uno de sus 95 ID; el score sólo agrega su balance neto en la muestra pública. Tres bloques perdieron -5, -14 y -5 aciertos netos; C1 perdió -6, bajo N_pub=20.998. Recomendar priorizar futuros diseños con mayor ganancia marginal de rango/firmas y validación prequential de score, junto con mejora train-only del prior. La composición del subconjunto público sigue desconocida.
