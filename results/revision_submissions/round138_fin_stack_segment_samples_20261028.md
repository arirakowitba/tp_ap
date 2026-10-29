# Dos pruebas representativas del modelo fin_stack (28-10-2026)

Base exacta 48, SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`. Scores públicos informados: 59=.94576 (67 flips, aprox. −12 vs48), 59 vendedores nuevos=.94619 (26, aprox. −3), 59 vistos→new=.94590 (27, aprox. −9). El bloque restante de 14 vistos→used queda aprox. cero. Son ecuaciones agregadas, jamás etiquetas individuales.

La predicción pura 63 difiere de 48 en 1518 filas: 809 de vendedores vistos y 709 nuevos. La banda 59 de máxima confianza perdió; no recomendar 63 de golpe. Para medir señal complementaria con menor exposición, se tomaron 150 IDs por segmento del universo de 820 desacuerdos de bandas 60–62 nunca probados (467 vistos, 353 nuevos), seleccionados por hash SHA256 determinista. Los dos CSV son disjuntos entre sí y respecto de 59. El sample es representativo del pool por segmento, no fue elegido por score público ni por labels de test.

| Archivo | Cambios | Dirección | SHA256 | Library |
|---|---:|---|---|---|
| `probe_fin_stack_seen_unprobed150_sobre48.csv` | 150 vistos | 18→used, 132→new | `37a8c519abff07ae8c64e360001229f3f3bf6953301908b38a946327ed00c832` | `libfile_c87eabc0f8bc81919e75eaf41e2e98d1` |
| `probe_fin_stack_new_unprobed150_sobre48.csv` | 150 nuevos | 63→used, 87→new | `ebb365337aa674d5ce099b763b4917089348539bf1b3b8e5f509e7950cfcf67a` | `libfile_5e872d5ed65c819183de873f7dfc76bf` |

Ambos tienen 30.000 IDs únicos y encabezado oficial. Código de reconstrucción: `submissions/20261028/build_fin_stack_segment_samples.py`.

Interpretación: enviar nuevo primero, luego visto si se decide usar ambos cupos para estimar heterogeneidad del modelo. Comparar cada score con 48; no inferir labels por ID. Un resultado positivo de una muestra no valida automáticamente el resto de la banda, porque el IC por vendedor puede ser amplio. Si ambas caen, no enviar 63 ni ampliar las bandas Claude; volver a entrenamientos con TE y stacks sin fuga, tune/confirm seller-group y bootstrap por vendedor. La mejor pública permanece 48=.94633 hasta que una submission la supere. No se enviaron archivos a Kaggle desde este trabajo.

## Resultado público de la muestra de vendedores nuevos

Ariel informó `probe_fin_stack_new_unprobed150_sobre48.csv` = **0,94495** frente a la base 48 = **0,94633**. Diferencia redondeada −0,00138, aproximadamente −29 aciertos si la partición pública tiene 21.000 filas. Corresponde al bloque conjunto de 150 flips (63→used, 87→new), SHA256 `ebb365337aa674d5ce099b763b4917089348539bf1b3b8e5f509e7950cfcf67a`. No identifica etiquetas individuales ni separa el saldo por dirección. El muestreo determinista del pool de 353 desacuerdos de vendedores nuevos de bandas 60–62 da evidencia desfavorable para ampliar ese segmento; no consolidar el bloque ni enviar 63 puro.

La muestra de 150 vendedores vistos es disjunta y sigue sin score. Su envío, sólo si se prioriza gastar el último cupo para medir otro segmento, evaluaría una hipótesis distinta; tiene 132 cambios →new y, por la pérdida previa de vistos→new en banda 59, el riesgo de caída es alto. La mejor pública sigue siendo 48=.94633.

## Cierre de las cinco pruebas públicas del día

Ariel informó `probe_fin_stack_seen_unprobed150_sobre48.csv` = **0,94257** (SHA256 `37a8c519abff07ae8c64e360001229f3f3bf6953301908b38a946327ed00c832`), frente a 48 = **0,94633**. Diferencia redondeada −0,00376, aproximadamente −79 aciertos si hay 21.000 filas públicas. Este bloque independiente tiene 150 flips de vendedores vistos, 18→used y 132→new. No se puede atribuir la caída a un ID ni a una dirección por separado.

Resumen de cinco CSV enviados por Ariel, todos sobre la base 48: 59 completa (67 flips) .94576, 59 nuevos (26) .94619, 59 vistos→new (27) .94590, bandas 60–62 muestra nuevos (150) .94495, bandas 60–62 muestra vistos (150) .94257. Los dos últimos bloques son disjuntos entre sí y de 59. Diferencias aproximadas respecto de 48, suponiendo 21.000 públicos: −12, −3, −9, −29 y −79, respectivamente. Los tres primeros se superponen y no se suman; los dos últimos son disjuntos, pero sus saldos agregados tampoco revelan etiquetas individuales. La mejor pública observada permanece 48=.94633.

Decisión: detener la expansión de `fin_stack` mediante bandas de margen, consenso o muestras 60–62. La CV aleatoria .9548/.1191 no trasladó su ventaja a estas pruebas públicas, especialmente en vendedores vistos. No enviar 60–63 como candidatos de explotación. Volver a comparar arquitectura, rasgos y procedencia de OOF frente a baselines con folds seller-group fit/tune/confirm, segmentación y bootstrap por vendedor antes de generar otra predicción.
