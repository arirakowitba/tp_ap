# Auditoría de identificabilidad de sondas públicas — 2026-11-02

Fuente: `results/intensive_20261030/submission_ledger_04_current.json` en Antigravity HEAD `197cd0d4b20bb43bc098c95371761fd441606170`. Lectura y cómputo local de patrones de intervención; no se ajustó ninguna etiqueta con scores.

- 73 entradas con score y mapa de flips, 64 vectores distintos (9 repeticiones). El ledger sigue incompleto respecto de las casi 100 submissions: faltan resultados recientes del 01-11 y probablemente archivos históricos.
- Unión de IDs intervenidos: 2.432 de los 30.000 test. Los restantes 27.568 no reciben una intervención registrada. Hay 381 firmas de pertenencia a sondas entre los intervenidos; sólo 216 firmas tienen un único ID y la mayor clase indistinguible contiene 606 IDs.
- La matriz de intervenciones tiene a lo sumo rango 64 por los 64 vectores distintos, frente a 2.432 IDs intervenidos. Incluso una firma única no garantiza identificar su etiqueta: las columnas pueden depender linealmente entre sí. La pertenencia al subconjunto público es desconocida y el score publicado está redondeado; ambos reducen todavía más la información identificable. No hay base para presentar probabilidades por ID como inferencia empírica calibrada de los scores.
- El reporte prequential anterior declara 48 pasos, pero su RMSE/accuracy direccional debe compararse contra baselines *en los mismos pasos*, separado por tamaño de intervención y excluyendo empates. Algunos errores de magnitud en pruebas recientes (D1, D2 y D1+D2) superan al predictor de cambio cero pese a acertar muchas direcciones de pruebas grandes negativas.
- Mejor score demostrado: D1+D2 0.94652. Las sondas posteriores reportadas (top22 .94623, incremento23 .94642, unánimes .94628, modelo completo OptionA .93695, semántica4 .94638) no desplazan el ancla.

## Diseño para las siguientes pruebas

1. Completar el ledger con hash de archivo completo, score y mapa de cambios *contra la misma ancla*; verificar IDs, polaridad y solapamientos. Marcar explícitamente `unresolved` cuando falte un CSV o el score.
2. Usar predicciones OOF con etiquetas train para establecer priors y calibración por segmento. Los scores públicos sólo actualizan parámetros **agregados de grupos diseñados de antemano**, con incertidumbre por redondeo, fracción pública y dependencia entre sondas. Backtest temporal con cada score retenido antes de usarlo.
3. Diseñar cuatro bloques pequeños y distinguibles a partir de desacuerdos de un candidato nuevo contra D1+D2, separados por vendedor visto/nuevo y confianza OOF; registrar la matriz de solapamiento. Congelar selección antes de mirar esos cuatro scores. Mantener un quinto CSV adaptativo de mejor score, condicionado al resultado conjunto, con respaldo exacto D1+D2 si la evidencia es insuficiente.
4. No seleccionar flips individuales por supuesta etiqueta posterior extraída de una sonda que los incluye junto a otros. Evitar reensayar firmas ya repetidas; favorecer bloques que separen clases de equivalencia y contrasten señales de modelos con validación pareada.

Límite: la cardinalidad pública se estima en 20.998, no se observan máscaras ni etiquetas públicas. Esta auditoría no predice rendimiento futuro ni promociona el candidato de texto de 660 cambios.
