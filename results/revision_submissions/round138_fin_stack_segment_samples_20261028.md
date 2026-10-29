# Dos pruebas representativas del modelo fin_stack (28-10-2026)

Base exacta 48, SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`. Scores públicos informados: 59=.94576 (67 flips, aprox. −12 vs48), 59 vendedores nuevos=.94619 (26, aprox. −3), 59 vistos→new=.94590 (27, aprox. −9). El bloque restante de 14 vistos→used queda aprox. cero. Son ecuaciones agregadas, jamás etiquetas individuales.

La predicción pura 63 difiere de 48 en 1518 filas: 809 de vendedores vistos y 709 nuevos. La banda 59 de máxima confianza perdió; no recomendar 63 de golpe. Para medir señal complementaria con menor exposición, se tomaron 150 IDs por segmento del universo de 820 desacuerdos de bandas 60–62 nunca probados (467 vistos, 353 nuevos), seleccionados por hash SHA256 determinista. Los dos CSV son disjuntos entre sí y respecto de 59. El sample es representativo del pool por segmento, no fue elegido por score público ni por labels de test.

| Archivo | Cambios | Dirección | SHA256 | Library |
|---|---:|---|---|---|
| `probe_fin_stack_seen_unprobed150_sobre48.csv` | 150 vistos | 18→used, 132→new | `37a8c519abff07ae8c64e360001229f3f3bf6953301908b38a946327ed00c832` | `libfile_c87eabc0f8bc81919e75eaf41e2e98d1` |
| `probe_fin_stack_new_unprobed150_sobre48.csv` | 150 nuevos | 63→used, 87→new | `ebb365337aa674d5ce099b763b4917089348539bf1b3b8e5f509e7950cfcf67a` | `libfile_5e872d5ed65c819183de873f7dfc76bf` |

Ambos tienen 30.000 IDs únicos y encabezado oficial. Código de reconstrucción: `submissions/20261028/build_fin_stack_segment_samples.py`.

Interpretación: enviar nuevo primero, luego visto si se decide usar ambos cupos para estimar heterogeneidad del modelo. Comparar cada score con 48; no inferir labels por ID. Un resultado positivo de una muestra no valida automáticamente el resto de la banda, porque el IC por vendedor puede ser amplio. Si ambas caen, no enviar 63 ni ampliar las bandas Claude; volver a entrenamientos con TE y stacks sin fuga, tune/confirm seller-group y bootstrap por vendedor. La mejor pública permanece 48=.94633 hasta que una submission la supere. No se enviaron archivos a Kaggle desde este trabajo.
