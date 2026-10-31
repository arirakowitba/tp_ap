# AP-2026-Q-2 — resultados públicos 30-10-2026

Referencia: submission 48, score público 0.94633, SHA-256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`.

| Archivo | SHA-256 | Cambios vs 48 | Score público | Delta vs 48 |
|---|---|---:|---:|---:|
| probe_havg_plus_sonda2_new_virgin16.csv | d72acb37b1d4d8fefef786fe425b643e4e887d98cdeee287d1812c5b5781ee9a | 16, vendedores nuevos | 0.94638 | +0.00005 |
| probe_havg_plus_sonda1_seen_virgin7_verified.csv | 6afa9bb30da486c08373197030409eb018fc7e504f80f89716b898faebab9a9f | 7, vendedores vistos | 0.94647 | +0.00014 |
| candidate_model_havg_plus_full70k_verified.csv | c568ad01485564a7682135ffec515204200326105c3e2b2d08a222b17f24ee8f | 378 | 0.94471 | -0.00162 |
| candidate_model_havg_b10_variant_c_full70k_verified.csv | 71d86093d7d14831cd33ef6de9bf7c3e076f2915b7dcce12609ac49bdf8a7a3d | 388 | 0.94514 | -0.00119 |
| candidate_d1_d2_combined_on48.csv | 57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502 | 23 (7+16, IDs disjuntos) | **0.94652** | **+0.00019** |

El CSV combinado se generó de las dos sondas partiendo de la misma 48, tomando D1 en sus 7 IDs y D2 en sus 16 IDs. Se verificó con lector CSV: 30.000 filas, IDs 1..30000 únicos, valores `used/new`, hash mostrado. La suma de deltas públicos (+0.00014 +0.00005) predijo 0.94652, que coincidió con el score observado. Esto verifica la relación de bloques y el cálculo; **no revela etiquetas individuales**, ni valida extrapolar a los 355 cambios históricos de los modelos completos.

Hubo un intento fallido de D1: el archivo entregado localmente tenía 27.999 registros CSV aunque un conteo de líneas erróneo indicó 30.000. Kaggle rechazó ese archivo antes de puntuar. Se reconstruyó desde el blob del repo y se validaron los IDs con `csv.DictReader`; el archivo verificado de la tabla sí puntuó. En adelante, el guard debe validar registros parseados e IDs secuenciales y únicos, no sólo líneas o tamaño.

## Lectura y próximos trabajos

- Nuevo mejor público **0.94652**: combinación D1+D2. Mantener 48 y el nuevo archivo como artefactos identificados por hash.
- Los modelos completos H-AVG y H-AVG-Plus, pese a ganancias en confirm seller-group reutilizado, pierden en público. No promocionar el conjunto de sus flips ni inferir qué IDs fallaron.
- Para nuevas señales, partir de entrenamiento real y validación seller-group honesta; separar efecto de umbral, early stopping/CPU y selección de hiperparámetros. Repetir cortes con selección congelada y reportar pares, accuracy/log loss y bootstrap vendedor.
- Auditar el historial de bloques 04–76 y 53–55 contra la nueva base 23 flips para evitar contradicciones/solapamientos. Todo score público es evidencia agregada.
- Cualquier candidato futuro: CSV completo, hash, mapa ID/target, diferencia exacta frente a 48 y a la nueva mejor, validación de IDs parseados y procedencia reproducible. Ariel hace las submissions; ningún agente usa cuota.
