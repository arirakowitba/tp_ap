# Cuatro cupos tras la regresión de 53

Referencia exacta 48=.94633, SHA256 `3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722`. 53=.94114, SHA256 `a894af2ebabbe3166caf984842813b1a47d0504b181a5800198655e2b15813a8`. Diferencia agregada −.00519; no hay etiquetas por ID. Los tres bloques 54/55/56 son disjuntos y su unión reproduce todos los 411 cambios de 53. Se definen antes de ver sus scores.

1. **54, primero**: 204 cambios de vendedor visto donde 53 coincide con HistGB+FastText y CatBoost. CSV SHA256 `5ac51b6926ac2d39264ae450bdc8bd8e0f99948ebf0ee893069bce04137debd5`.
2. **55, después del score 54**: 65 cambios que coinciden con HistGB+FastText y discrepan de CatBoost. CSV SHA256 `fb82ba0afec97a166d4744cadd8967e028b46824204e67adac8336d8f9711eb5`.
3. **56, después de evaluar los dos bloques anteriores**: 142 cambios que discrepan de HistGB+FastText; 126/142 también discrepan de CatBoost. CSV SHA256 `1ca4da090e29e1b60bb8bebaf09d97fb0211b35e9e85ada4851487c8178e36b6`. Menor prioridad por desacuerdo entre modelos, pero podría aislar un efecto positivo oculto en el saldo malo de 53.
4. Reservar cuarto cupo para una ampliación o contracción *predeclarada* del bloque ganador, o una nueva familia con validación train, después de recibir los scores. No reenviar 48 ni 53 sin cambios.

Cada score actualiza el efecto del grupo completo en el subconjunto público, nunca labels individuales. La relación aditiva aproximada con 53 puede fallar a cinco decimales por redondeo y por cualquier cambio de evaluación. Los bloques 54–56 se basan en diagnóstico de test y modelos ya entrenados; no son evidencia de mejora esperada frente a 48. La selección de 54 como primer experimento busca mayor apoyo de modelos y valor informativo. Ningún envío es realizado por el asistente.

Reproducción: `round122_partition53.py` consume CSV exactos 48/53, probabilidades guardadas de HistGB+FastText y CatBoost y emite `round122_changed_rows.csv` con ID, dirección, segmento, seller_count y probabilidades; `round122_manifest.json` incluye hashes y tamaños. No consume etiquetas de test ni deriva estados individuales desde el leaderboard.

Library: 54 `libfile_3208460aae608191bb598624b81ee4ba`; 55 `libfile_220f8ec2f4648191bee01543493f7e0a`; 56 `libfile_d095940501c481919d24f7d4e0872f1d`; checkpoint `libfile_fa6b72bd28e881919ae7bd0d05db6bc4`.
