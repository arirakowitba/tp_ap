# Pseudoetiquetas seguras de test para un profesor de texto

Se ajustaron TF-IDF word 1–2 y char_wb 2–5 con el train de cada corte agrupado por vendedor, cada uno con regresión logística C=2. Se seleccionaron filas de test solo si ambos profesores asignaban la misma clase con probabilidad >= umbral. Se reentrenó un profesor conjunto con pesos 0.25 o 1 para las pseudoetiquetas. Ninguna etiqueta del corte de evaluación entró en la selección, vectorización o ajuste. Validación exclusivamente con etiquetas reales de train. Los cortes 883 y 2026 se solapan, por lo que no son confirmaciones independientes.

| Umbral | Pseudo 883 | Errores high-confidence holdout 883 | Pseudo 2026 | Errores holdout 2026 | Δ aciertos peso 0.25 (883 / 2026) | Δ aciertos peso 1 (883 / 2026) |
|---:|---:|---:|---:|---:|---:|---:|
| .95 | 4271 | 16 / 1731 | 4512 | 30 / 1670 | +1 / +2 | -2 / -1 |
| .97 | 2472 | 8 / 979 | 2714 | 13 / 1000 | +1 / +1 | +1 / +2 |
| .99 | 606 | 0 / 230 | 672 | 2 / 227 | +1 / +1 | +2 / +1 |
| .995 | 212 | 0 / 81 | 255 | 1 / 98 | +2 / +1 | +2 / +1 |

El profesor textual por sí solo está en .84583 / .83851, lejos del HistGB combinado. Las mejoras de pseudoentrenamiento son demasiado pequeñas para generar submission. Los 8.982 casos de test en que HistGB y CatBoost coinciden con probabilidad >.99 para la clase elegida también coinciden **todos** con la submission 33. Para umbral .995 son 5.757. Por lo tanto, esta familia no descubre flips obvios contra 33: los desacuerdos relevantes están en probabilidades menos extremas, donde validar bien importa más.

No tratar las pseudoetiquetas como ground truth ni como validación adicional; usarlas solo para ajustar modelos con pesos y reservar train real para medir. Los JSON del checkpoint contienen parámetros y resultados. Siguiente hipótesis: arquitectura/tabular que modele el residual de desacuerdos cerca del umbral, con splits de vendedor y confirmación independiente; no seguir subiendo el umbral para buscar cambios inexistentes.
