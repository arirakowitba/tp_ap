# Recalculo robusto de la validación cronológica AG-0055

**Fecha:** 2026-11-04  
**Fuente:** `results/adversarial_posterior_audit_20261104/chronological_holdout_recent_probes.csv` en AG HEAD `4c20a6e8bf4159e550ee6394f82b5282e69a66f2`.

## Resultados reproducidos

| Evaluación | n | MAE hits | RMSE hits | Pearson r | Sign accuracy (reales no cero) |
|---|---:|---:|---:|---:|---:|
| 20 filas completas | 20 | 4.6403 | 7.9927 | 0.994783 | 66.67% |
| Sin `candidate_optionA...` (delta -197) | 19 | 3.3271 | 4.5995 | **-0.013356** | 64.71% |
| Cinco probes 20261104 | 5 | 3.4896 | 4.0311 | 0.484251 | 75.00% (3/4 no nulos) |

## Predicciones guardadas para los probes 20261104

| Probe | Delta real vs 48 | Predicción guardada | Signo |
|---|---:|---:|---|
| all80 | 0 | -5.1781 | no evaluable |
| used40 | -2 | -1.0384 | correcto |
| new40 | +6 | **-0.2224** | **incorrecto** |
| low20 | +6 | +2.4928 | correcto |
| low10 | +2 | +3.5786 | correcto |

## Conclusión

La correlación 0.9948 no es evidencia de transporte: está dominada por un único candidato de gran magnitud. Sin ese punto, la correlación es esencialmente cero. El reporte AG-0055 afirma 100% de signo en los cinco probes usando números que no coinciden con el CSV versionado; el artefacto reproducible da 75% sobre los cuatro deltas no nulos.

El posterior puede conservarse como herramienta exploratoria, pero no debe producir probabilidades calibradas, ganancias esperadas ni decisiones de gasto hasta:

1. fijar el conjunto y orden cronológico con timestamps verificados;
2. separar familias por linaje/overlap real;
3. reportar métricas robustas con y sin outliers;
4. validar prospectivamente nuevas familias;
5. mantener scores agregados como restricciones de grupo.

## Aclaración sobre la matriz de incidencia

Para clasificación binaria, una matriz 0/1 de toggles sí puede representar el efecto respecto de una referencia común, aunque no guarde dirección explícita, **si y sólo si** cada vector está canónicamente reconstruido contra el mismo archivo base exacto. Por lo tanto, la corrección exigida no es añadir signo arbitrariamente: es demostrar la canonicalización hash-a-hash frente a Sub48 y excluir cualquier entrada cuyo baseline o vector no pueda verificarse. El ledger tiene 85 entradas `verified_recoverable`, pero diez carecen de `flips_dict`; requieren reconstrucción fail-closed antes de usarlas para generar targets o candidates.
