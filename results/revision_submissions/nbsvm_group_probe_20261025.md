# NB-SVM sobre textos originales, cortes agrupados por vendedor

Fuente: original_docs.pkl derivado del Sheet original 100k; etiquetas solo de train. Corte 883: 55.665 ajuste / 14.335 validación. Corte 2026: 56.507 ajuste / 13.493 validación. En cada corte, TF-IDF word 1-2 se ajustó únicamente con train; ratio log NB supervisado y regresión logística C=1. Mezcla fija con el HistGB del corte y umbral histórico 0.475.

| Corte | HistGB | NB-SVM solo | Mezcla 10% NB-SVM | Neto / cambios | Mezcla 20% |
|---|---:|---:|---:|---:|---:|
| 883 | 0.928776 | 0.841856 | 0.928985 | +3 / 27 | 0.928636 |
| 2026 | 0.918254 | 0.835100 | 0.918180 | -1 / 41 | 0.918180 |

El primer corte sirvió para elegir peso 0.1; el segundo no confirmó ganancia. Ambos cortes pueden solaparse en etiquetas y no constituyen una confirmación completamente independiente. NB-SVM no merece submission ni refit test como señal aislada. El experimento no usó Kaggle ni etiquetas de leaderboard. Reproducir con nbsvm_group_probe.py y el checkpoint 30, que incluye original_docs.pkl y los NPZ HistGB.
