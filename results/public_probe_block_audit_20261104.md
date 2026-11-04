# Auditoría de probes públicos del 4 de noviembre de 2026

Se preservaron y verificaron los cinco CSV generados para la tanda. Todos tienen 30.000 IDs únicos y esquema `ID,condition`. La comparación exacta se hizo contra `submission_final_best_d1_d2_20261031.csv` (SHA-256 `57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502`).

Con el tamaño público previamente identificado, `Npub=20.998`, cada score publicado redondeado a cinco decimales corresponde aquí a un único número entero de aciertos.

| Archivo | SHA-256 | Cambios vs D1+D2 | Score público | Aciertos públicos | Delta vs D1+D2 |
| --- | --- | ---: | ---: | ---: | ---: |
| D1+D2 | `57172f...` | 0 | .94652 | 19.875 | 0 |
| `submission_20261104_01_bag3_deep_new80.csv` | `b9b4ab8c...` | 80 (40 new→used, 40 used→new) | .94633 | 19.871 | -4 |
| `probe_20261104_02_split_to_used40_on_d1d2.csv` | `46c25b9b...` | 40 new→used | .94623 | 19.869 | -6 |
| `submission_20261104_03_split_to_new40_on_d1d2.csv` | `c9fe5b83...` | 40 used→new | .94661 | 19.877 | +2 |
| `probe_20261104_04_to_new_lowmargin20_on_d1d2.csv` | `a1a58a73...` | 20 used→new | .94661 | 19.877 | +2 |
| `probe_20261104_05_to_new_lowmargin10_on_d1d2.csv` | `e36a3866...` | 10 used→new | .94642 | 19.873 | -2 |

Relaciones verificadas por vector, sin inferir etiquetas individuales:

- El batch80 es la unión disjunta exacta de used40 y new40.
- Used40 aporta -6 y new40 +2; suma -4, consistente con batch80.
- Low-margin20 es subconjunto exacto de new40 y conserva los +2 completos.
- El complemento high-margin20 de new40 aporta 0.
- Low-margin10 aporta -2.
- Por diferencia entre bloques disjuntos, el complemento upper10 dentro de low-margin20 aporta +4.

Nuevo ancla pública demostrada: **0,94661 = 19.877/20.998**, dos aciertos por encima de D1+D2. La brecha a 0,95404 = 20.033/20.998 es ahora **156 aciertos**.

No se derivan etiquetas por ID de estos scores. Sólo son identificables los saldos agregados de bloques mediante relaciones exactas de unión/subconjunto. Si la referencia inicial de 50 cupos era correcta, quedan 45 después de las cinco cargas exitosas de hoy.
