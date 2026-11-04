# Auditoría del prior fold-específico en target encoding OOF (2026-11-03)

Se auditó el `make_te` usado por los experimentos nuevos de CatBoost/LightGBM en el corte 20261144. La implementación histórica suaviza cada mapa OOF con `y[fit].mean()`; ese prior contiene etiquetas del fold que se está validando. La alternativa limpia usa `y[fit[train_fold]].mean()` dentro de cada fold. Tune y confirm siguen usando únicamente estadísticas de fit.

- Split: fit 45.294, tune 12.442, confirm 12.264.
- Prior global de fit: 0,462313.
- Priors de entrenamiento de los cuatro folds: 0,467147; 0,461319; 0,459981; 0,460805.
- Las 45.294 filas OOF cambian en las tres columnas auditadas.
- `category_id`: cambio absoluto medio 0,001235; p95 0,004029; máximo 0,004835.
- `listing_type_id`: cambio medio 0,00000436; máximo 0,001458.
- `state`: cambio medio 0,0000140; máximo 0,004029.

Esto demuestra una fuga de etiqueta pequeña pero real en el prior. No cuantifica por sí solo el efecto sobre accuracy: se requiere reentrenar el control CatBoost bag3 y el blend CatBoost/LightGBM con el prior fold-específico y comparar de forma pareada. Hasta esa réplica, los resultados de AG-0046/0047 deben considerarse desarrollo prometedor, no OOF estrictamente limpio.

Código: `scripts/audit_te_fold_prior_20261103.py`. Resultado máquina: `results/te_fold_prior_audit_20261103.json`. No se usaron scores públicos ni etiquetas de test y no se consumió cuota Kaggle.
